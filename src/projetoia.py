import os
import pymupdf
import json
import sqlite3
import base64
import uuid
import concurrent.futures
import threading
import time          
import pandas as pd
from openai import OpenAI
import chromadb 

from dash import Dash, html, dcc, dash_table, Input, Output, State, callback_context
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 1. CONFIGURAÇÕES E API
# ==========================================

client = OpenAI(
    api_key="",
    base_url="https://api.groq.com/openai/v1"
)

# Travas de segurança para os bancos de dados
db_lock = threading.Lock()
chroma_lock = threading.Lock()

# ==========================================
# 2. BANCOS DE DADOS (SQLITE E CHROMA)
# ==========================================

def criar_banco():
    conn = sqlite3.connect('curriculos.db')
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS candidatos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nome TEXT, email TEXT, telefone TEXT, cidade TEXT,
        linkedin TEXT, github TEXT, anos_experiencia INTEGER,
        score_geral INTEGER, nivel_profissional TEXT,
        skills TEXT, texto_completo TEXT,
        data_cadastro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

criar_banco()

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="curriculos_vetores")

def salvar_no_banco_relacional(candidato, texto_completo):
    conn = sqlite3.connect("curriculos.db", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO candidatos (
        nome, email, telefone, cidade, linkedin, github,
        anos_experiencia, score_geral, nivel_profissional, skills, texto_completo
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        candidato.get("nome"), candidato.get("email"), candidato.get("telefone"),
        candidato.get("cidade"), candidato.get("linkedin"), candidato.get("github"),
        candidato.get("anos_experiencia", 0), candidato.get("score_geral", 0),
        candidato.get("nivel_profissional"), json.dumps(candidato.get("skills", []), ensure_ascii=False),
        texto_completo
    ))
    conn.commit()
    conn.close()

def salvar_no_vetor(candidato, texto_bruto):
    nome = candidato.get("nome", "Desconhecido")
    skills_str = ", ".join([s["nome"] for s in candidato.get("skills", [])])
    
    texto_para_vetorizar = f"Candidato: {nome}. Experiência: {candidato.get('anos_experiencia')} anos. Nível: {candidato.get('nivel_profissional')}. Skills principais: {skills_str}. Resumo do currículo: {texto_bruto[:1500]}"
    
    collection.add(
        documents=[texto_para_vetorizar],
        metadatas=[{"nome": nome}],
        ids=[str(uuid.uuid4())] 
    )

# ==========================================
# 3. FUNÇÕES DE PROCESSAMENTO E IA (GROQ)
# ==========================================

def extrair_texto_pdf(caminho_pdf):
    texto = []
    with pymupdf.open(caminho_pdf) as pdf:
        for pagina in pdf:
            texto.append(pagina.get_text())
    return "\n".join(texto)

def calcular_nivel_skill(score):
    if score < 40: return "Iniciante"
    elif score < 60: return "Junior"
    elif score < 80: return "Pleno"
    elif score < 95: return "Senior"
    return "Expert"

def calcular_nivel_profissional(score_geral, anos_experiencia):
    if score_geral >= 90 and anos_experiencia >= 8: return "Expert"
    if score_geral >= 80 and anos_experiencia >= 4: return "Senior"
    if score_geral >= 70 and anos_experiencia >= 2: return "Pleno"
    if score_geral >= 50: return "Junior"
    return "Iniciante"

def normalizar_scores(skills):
    for skill in skills:
        score = skill.get("score", 0)
        try: score = float(score)
        except: score = 0
        if score <= 10: score *= 10
        score = max(0, min(100, round(score)))
        skill["score"] = score
        skill["nivel"] = calcular_nivel_skill(score)
    return skills

def calcular_score_geral(skills):
    if not skills: return 0
    scores = [skill["score"] for skill in skills if isinstance(skill.get("score"), (int, float))]
    if not scores: return 0
    return round(sum(scores) / len(scores))

def analisar_curriculo(caminho_pdf):
    texto_curriculo = extrair_texto_pdf(caminho_pdf)
    prompt = f"""
Analise o currículo abaixo. 
Formato OBRIGATÓRIO de saída (JSON puro):
{{
  "nome": "", "email": "", "telefone": "", "cidade": "", "linkedin": "", "github": "",
  "anos_experiencia": 0,
  "skills": [
    {{ "nome": "", "score": 0, "explicacao": "" }}
  ]
}}
Regras: Retorne até 10 skills, score entre 0 e 100, não invente informações.
Currículo:
{texto_curriculo}
"""
    max_tentativas = 5
    conteudo = None
    
    for tentativa in range(max_tentativas):
        try:
            import openai
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                temperature=0,
                response_format={"type": "json_object"}, 
                messages=[
                    {"role": "system", "content": "Você é um extrator de dados de RH. Responda OBRIGATORIAMENTE em formato JSON válido."},
                    {"role": "user", "content": prompt}
                ]
            )
            conteudo = response.choices[0].message.content
            break 
            
        except openai.RateLimitError as e: 
            if tentativa < max_tentativas - 1:
                espera = 8 
                time.sleep(espera)
            else:
                raise Exception(f"Falha após {max_tentativas} tentativas por limite de taxa da API.")
    
    try:
        candidato = json.loads(conteudo)
        skills = normalizar_scores(candidato.get("skills", []))
        candidato["skills"] = skills
        candidato["score_geral"] = calcular_score_geral(skills)
        candidato["nivel_profissional"] = calcular_nivel_profissional(candidato["score_geral"], candidato.get("anos_experiencia", 0))
        return candidato
    except Exception as e:
        raise e

# --- IMPLEMENTAÇÃO DO RAG NO MATCH DE VAGA ---
def analisar_vaga_com_candidatos(descricao_vaga, df_cand):
    if df_cand.empty:
        return {}, "Nenhum candidato no banco de dados para analisar."

    qnt_total_candidatos = len(df_cand)
    k_limite = min(10, qnt_total_candidatos)

    # 1. RAG RETRIEVAL
    resultados = collection.query(
        query_texts=[descricao_vaga],
        n_results=k_limite
    )
    
    nomes_filtrados_rag = [meta["nome"] for meta in resultados["metadatas"][0]]
    df_filtrado_rag = df_cand[df_cand['nome'].isin(nomes_filtrados_rag)]

    # 2. GENERATION
    resumo_candidatos = df_filtrado_rag[["nome", "anos_experiencia", "nivel_profissional", "skills"]].to_dict("records")
    
    prompt = f"""
    Você é um Tech Recruiter Sênior.
    Abaixo está a DESCRIÇÃO DA VAGA e uma lista pré-filtrada com os melhores CANDIDATOS.
    
    DESCRIÇÃO DA VAGA:
    "{descricao_vaga}"
    
    CANDIDATOS TOP SELECIONADOS:
    {json.dumps(resumo_candidatos, ensure_ascii=False, indent=2)}
    
    Sua tarefa é avaliar a aderência de CADA candidato listado acima à vaga.
    
    REGRAS DE PONTUAÇÃO (Siga estritamente para cada candidato):
    - 90 a 100: Candidato ideal. Requisitos obrigatórios e senioridade batem.
    - 70 a 89: Bom candidato. Faltam diferenciais ou a senioridade é menor.
    - 40 a 69: Tem bagagem técnica, mas a stack principal difere. Não zere a nota, valorize a base.
    - 0 a 39: Área totalmente diferente.
    
    FORMATO DE SAÍDA OBRIGATÓRIO (JSON):
    Retorne APENAS um JSON válido. Ele deve conter uma justificativa geral e um objeto 'avaliacoes' com os dados de cada candidato.
    {{
        "explicacao_geral": "Parágrafo curto justificando quem é o melhor candidato do grupo e por que.",
        "avaliacoes": {{
            "Nome Exato do Candidato 1": {{
                "score": 95,
                "raciocinio": "Sua justificativa clara e direta para esta nota."
            }},
            "Nome Exato do Candidato 2": {{
                "score": 60,
                "raciocinio": "Sua justificativa clara e direta..."
            }}
        }}
    }}
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.1, 
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Você é um especialista em RH Tech. Responda OBRIGATORIAMENTE em formato JSON válido."},
                {"role": "user", "content": prompt}
            ]
        )
        
        data = json.loads(response.choices[0].message.content)
        return data.get("avaliacoes", {}), data.get("explicacao_geral", "Análise concluída.")
        
    except Exception as e:
        return {}, f"Erro ao analisar candidatos: {str(e)}"

# ==========================================
# 4. FUNÇÃO ISOLADA PARA MULTITHREADING
# ==========================================
def processar_unico_pdf(conteudo, nome):
    caminho_temp = f"temp_{uuid.uuid4().hex[:8]}_{nome}"
    try:
        content_type, content_string = conteudo.split(',')
        decoded = base64.b64decode(content_string)
        
        with open(caminho_temp, "wb") as f:
            f.write(decoded)
        
        novo_candidato = analisar_curriculo(caminho_temp)
        texto_bruto = extrair_texto_pdf(caminho_temp)
        
        with db_lock:
            salvar_no_banco_relacional(novo_candidato, texto_bruto)
            
        with chroma_lock:
            salvar_no_vetor(novo_candidato, texto_bruto)
        
        os.remove(caminho_temp) 
        return {"sucesso": True, "nome": nome}
    except Exception as e:
        if os.path.exists(caminho_temp):
            os.remove(caminho_temp)
        print(f"Falha final em {nome}: {str(e)}")
        return {"sucesso": False, "nome": nome, "erro": str(e)}

# ==========================================
# 5. APLICAÇÃO DASH E LAYOUT
# ==========================================

def gerar_grafico_vazio(titulo):
    fig = go.Figure()
    fig.update_layout(
        title=titulo, height=320, template="plotly_dark",
        xaxis={"visible": False}, yaxis={"visible": False},
        annotations=[{"text": "Aguardando currículos...", "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 14, "color": "#777"}}],
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

app = Dash(__name__)

app.layout = html.Div(
    style={
        "backgroundColor": "#121212", 
        "minHeight": "100vh", "margin": "0", "padding": "30px",
        "color": "#E0E0E0", "fontFamily": "Inter, sans-serif"
    },
    children=[
        dcc.Store(id='store-scores', data=None),

        html.Div([
            html.H1("Dashboard RH", style={"margin": "0", "color": "#FFFFFF"}),
            html.P("Busca semântica inteligente e explicável.", style={"color": "#A0A0A0"})
        ], style={"marginBottom": "30px", "textAlign": "center"}),

        html.Div(id="kpi-cards", style={"display": "flex", "flexWrap": "wrap", "gap": "20px", "marginBottom": "30px"}),

        html.Div([
            html.Div([
                html.H3("Adicionar Currículos (Em Lote)", style={"marginTop": "0"}),
                dcc.Upload(
                    id='upload-pdf',
                    children=html.Div(['Arraste ou ', html.A('Selecione os PDFs')]),
                    style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '10px', 'textAlign': 'center', 'borderColor': '#5C6BC0', 'cursor': 'pointer', 'backgroundColor': '#1E1E1E'},
                    multiple=True 
                ),
                html.Div(id="upload-status", style={"marginTop": "15px", "textAlign": "center"})
            ], style={"flex": "1", "minWidth": "300px", "padding": "20px", "backgroundColor": "#1E1E1E", "borderRadius": "15px", "boxShadow": "0 4px 6px rgba(0,0,0,0.3)"}),

            html.Div([
                html.H3("Busca Vetorial da Melhor Vaga", style={"marginTop": "0"}),
                dcc.Textarea(
                    id='input-vaga',
                    placeholder="Cole a descrição da vaga aqui (ex: Preciso de um dev React Pleno com Docker...)",
                    style={"width": "100%", "height": "80px", "borderRadius": "8px", "padding": "10px", "backgroundColor": "#2A2A2A", "color": "white", "border": "none"}
                ),
                html.Button(
                    "Analisar e Recalcular Score", 
                    id="btn-match", 
                    n_clicks=0,
                    style={"marginTop": "10px", "backgroundColor": "#5C6BC0", "color": "white", "border": "none", "padding": "10px 20px", "borderRadius": "8px", "cursor": "pointer", "fontWeight": "bold", "width": "100%"}
                ),
            ], style={"flex": "1", "minWidth": "300px", "padding": "20px", "backgroundColor": "#1E1E1E", "borderRadius": "15px", "boxShadow": "0 4px 6px rgba(0,0,0,0.3)"})
        ], style={"display": "flex", "flexWrap": "wrap", "gap": "20px", "marginBottom": "30px"}),

        html.Div(id="resultado-match", style={"display": "none"}),

        html.Div([
            html.Div([dcc.Graph(id="grafico_niveis")], style={"flex": "1", "minWidth": "300px", "backgroundColor": "#1E1E1E", "borderRadius": "15px", "padding": "15px", "boxShadow": "0 4px 6px rgba(0,0,0,0.3)", "overflow": "hidden"}),
            html.Div([dcc.Graph(id="grafico_skills")], style={"flex": "2", "minWidth": "300px", "backgroundColor": "#1E1E1E", "borderRadius": "15px", "padding": "15px", "boxShadow": "0 4px 6px rgba(0,0,0,0.3)", "overflow": "hidden"})
        ], style={"display": "flex", "flexWrap": "wrap", "gap": "20px", "marginBottom": "30px"}),

        html.Div([
            html.Button("Limpar Filtros e Scores", id="btn_limpar", n_clicks=0, style={"backgroundColor":"#dc3545", "color":"white", "border":"none", "padding":"10px 20px", "borderRadius":"8px", "cursor":"pointer"})
        ], style={"marginBottom": "20px"}),

        html.Div([
            dash_table.DataTable(
                id="tabela_candidatos",
                page_size=10,
                sort_action="native",
                style_table={"overflowX": "auto", "borderRadius": "10px"},
                style_header={"backgroundColor": "#333", "color": "white", "fontWeight": "bold", "border": "none"},
                style_cell={
                    "backgroundColor": "#222", "color": "#E0E0E0", 
                    "textAlign": "left", "padding": "15px", "borderBottom": "1px solid #444",
                    "whiteSpace": "normal", 
                    "height": "auto",
                    "maxWidth": "350px" 
                }
            )
        ], style={"backgroundColor": "#1E1E1E", "padding": "20px", "borderRadius": "15px", "boxShadow": "0 4px 6px rgba(0,0,0,0.3)"})
    ]
)

# ==========================================
# 6. CALLBACK GIGANTE
# ==========================================

@app.callback(
    [Output("tabela_candidatos", "data"), 
     Output("tabela_candidatos", "columns"),
     Output("grafico_niveis", "figure"), 
     Output("grafico_skills", "figure"),
     Output("kpi-cards", "children"), 
     Output("upload-status", "children"),
     Output("resultado-match", "children"),
     Output("resultado-match", "style"),
     Output("store-scores", "data")],
    [Input("upload-pdf", "contents"), 
     Input("btn_limpar", "n_clicks"), 
     Input("grafico_niveis", "clickData"), 
     Input("grafico_skills", "clickData"),
     Input("btn-match", "n_clicks")],
    [State("upload-pdf", "filename"), 
     State("upload-status", "children"),
     State("input-vaga", "value"),
     State("store-scores", "data"),
     State("resultado-match", "children"),
     State("resultado-match", "style")] 
)
def atualizar_dashboard(conteudos_pdf, n_limpar, click_nivel, click_skill, n_match, nomes_arquivos, status_atual, texto_vaga, dict_scores_memoria, match_children, match_style):
    ctx = callback_context
    trigger = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    msg_upload = status_atual 
    novo_match_children = match_children
    novo_match_style = match_style if match_style else {"display": "none"}
    novos_scores_memoria = dict_scores_memoria
    
    if trigger == "btn_limpar":
        novos_scores_memoria = None 
        novo_match_children = None
        novo_match_style = {"display": "none"}

    if trigger == "upload-pdf" and conteudos_pdf is not None:
        if not isinstance(conteudos_pdf, list):
            conteudos_pdf = [conteudos_pdf]
            nomes_arquivos = [nomes_arquivos]
            
        resultados = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(processar_unico_pdf, c, n) for c, n in zip(conteudos_pdf, nomes_arquivos)]
            for future in concurrent.futures.as_completed(futures):
                resultados.append(future.result())
                
        teve_erro = any(not r["sucesso"] for r in resultados)
        msg_upload = html.Div("Alguns apresentaram erro.", style={"color": "#FFC107", "fontWeight": "bold"}) if teve_erro else html.Div("Arquivos processados!", style={"color": "#4CAF50", "fontWeight": "bold"})

    conn = sqlite3.connect("curriculos.db")
    df = pd.read_sql("SELECT * FROM candidatos", conn)
    conn.close()

    if trigger == "btn-match":
        if not texto_vaga:
            novo_match_children = html.Div([html.H4("Aviso"), html.P("Por favor, insira a descrição da vaga antes de analisar.")])
            novo_match_style = {"padding": "20px", "backgroundColor": "#D32F2F", "borderRadius": "10px", "marginBottom": "30px", "color": "white", "display": "block"}
        else:
            novos_scores_memoria, explicacao = analisar_vaga_com_candidatos(texto_vaga, df)
            novo_match_children = html.Div([html.H4("Melhor Escolha:", style={"marginTop": "0"}), html.P(explicacao)])
            novo_match_style = {"padding": "20px", "backgroundColor": "#155724", "border": "1px solid #c3e6cb", "borderRadius": "10px", "marginBottom": "30px", "color": "#d4edda", "display": "block"}

    if novos_scores_memoria and not df.empty:
        scores_map = {k: v.get("score", 0) for k, v in novos_scores_memoria.items()}
        raciocinios_map = {k: v.get("raciocinio", "Sem justificativa") for k, v in novos_scores_memoria.items()}
        
        texto_reprovado_rag = "Barrado na Triagem Inicial: O perfil não apresentou palavras-chave ou contexto semântico suficientes em relação aos requisitos desta vaga específica para avançar para a análise profunda da IA."
        
        df['score_geral'] = df['nome'].map(scores_map).fillna(25).astype(int)
        df['justificativa_ia'] = df['nome'].map(raciocinios_map).fillna(texto_reprovado_rag) 
        df = df.sort_values(by="score_geral", ascending=False)
    else:
        df['justificativa_ia'] = "-" 

    df_filtrado = df.copy()
    if trigger != "btn_limpar":
        if click_nivel:
            nivel = click_nivel["points"][0]["label"]
            df_filtrado = df_filtrado[df_filtrado["nivel_profissional"] == nivel]
        if click_skill:
            skill = click_skill["points"][0]["x"]
            def check_skill(row_skills):
                try: return any(s.get("nome") == skill for s in json.loads(row_skills))
                except: return False
            df_filtrado = df_filtrado[df_filtrado["skills"].apply(check_skill)]

    colunas_tabela = ["nome", "email", "cidade", "anos_experiencia", "score_geral", "nivel_profissional"]
    
    if novos_scores_memoria:
        colunas_tabela.append("justificativa_ia")

    nome_coluna_score = "Score Geral" if not novos_scores_memoria else "Score da Vaga"
    
    dados_tabela = df_filtrado[colunas_tabela].to_dict("records") if not df_filtrado.empty else []
    
    cols = []
    for c in colunas_tabela:
        if c == "score_geral":
            titulo = nome_coluna_score
        elif c == "justificativa_ia":
            titulo = "Raciocínio da IA"
        else:
            titulo = c.replace("_", " ").title()
        cols.append({"name": titulo, "id": c})

    total = len(df)
    score_medio = round(df["score_geral"].mean(), 1) if not df.empty else 0
    titulo_kpi_score = "Score Médio (Geral)" if not novos_scores_memoria else "Score Médio (Baseado na Vaga)"

    estilo_kpi = {"flex": "1", "minWidth": "200px", "padding": "20px", "background": "#1E1E1E", "borderRadius": "15px", "textAlign": "center", "boxShadow": "0 4px 6px rgba(0,0,0,0.3)", "borderTop": "4px solid #5C6BC0"}
    kpis = [
        html.Div([html.H2(total, style={"margin": "0"}), html.P("Total de Currículos")], style=estilo_kpi),
        html.Div([html.H2(score_medio, style={"margin": "0"}), html.P(titulo_kpi_score, style={"color": "#FFD54F" if novos_scores_memoria else "#A0A0A0"})], style=estilo_kpi),
    ]

    if df.empty:
        fig_niveis = gerar_grafico_vazio("Senioridade")
        fig_skills = gerar_grafico_vazio("Top Skills Cadastradas")
    else:
        nivel_df = df["nivel_profissional"].value_counts().reset_index()
        nivel_df.columns = ["nivel", "quantidade"]
        fig_niveis = px.pie(nivel_df, names="nivel", values="quantidade", hole=0.5, template="plotly_dark", title="Senioridade")
        fig_niveis.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

        skills = []
        for item in df["skills"]:
            try:
                for s in json.loads(item): skills.append(s["nome"])
            except: pass
        skills_df = pd.DataFrame(skills, columns=["skill"])
        skills_df = skills_df["skill"].value_counts().head(10).reset_index()
        skills_df.columns = ["skill", "quantidade"]
        
        fig_skills = px.bar(skills_df, x="skill", y="quantidade", title="Top Skills Cadastradas", template="plotly_dark")
        fig_skills.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

    return dados_tabela, cols, fig_niveis, fig_skills, kpis, msg_upload, novo_match_children, novo_match_style, novos_scores_memoria

if __name__ == "__main__":
    app.run(port=8050)