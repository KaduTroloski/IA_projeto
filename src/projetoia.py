import os
import pymupdf
import json
import re
import sqlite3
import base64
import pandas as pd
from openai import OpenAI

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

# ==========================================
# 2. BANCO DE DADOS
# ==========================================

def criar_banco():
    conn = sqlite3.connect('curriculos.db')
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS candidatos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nome TEXT,
        email TEXT,
        telefone TEXT,
        cidade TEXT,
        linkedin TEXT,
        github TEXT,
        anos_experiencia INTEGER,
        score_geral INTEGER,
        nivel_profissional TEXT,
        skills TEXT,
        texto_completo TEXT,
        data_cadastro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

# Garante que o banco exista ao iniciar o app
criar_banco()

def salvar_no_banco(candidato, texto_completo):
    conn = sqlite3.connect("curriculos.db")
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO candidatos (
        nome, email, telefone, cidade, linkedin, github,
        anos_experiencia, score_geral, nivel_profissional, skills, texto_completo
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        candidato.get("nome"),
        candidato.get("email"),
        candidato.get("telefone"),
        candidato.get("cidade"),
        candidato.get("linkedin"),
        candidato.get("github"),
        candidato.get("anos_experiencia", 0),
        candidato.get("score_geral", 0),
        candidato.get("nivel_profissional"),
        json.dumps(candidato.get("skills", []), ensure_ascii=False),
        texto_completo
    ))
    conn.commit()
    conn.close()

# ==========================================
# 3. FUNÇÕES DE PROCESSAMENTO E IA (GROQ)
# ==========================================

def extrair_texto_pdf(caminho_pdf):
    texto = []
    with pymupdf.open(caminho_pdf) as pdf:
        for pagina in pdf:
            texto.append(pagina.get_text())
    return "\n".join(texto)

def limpar_json(resposta):
    resposta = resposta.strip()
    resposta = re.sub(r"^```json\s*|\s*```$", "", resposta, flags=re.MULTILINE)
    return resposta.strip()

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
        try:
            score = float(score)
        except:
            score = 0
        if score <= 10:
            score *= 10
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
Retorne APENAS JSON válido.
NÃO utilize markdown.
NÃO utilize ```json.
NÃO escreva explicações.

Formato obrigatório:
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
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0,
        messages=[
            {"role": "system", "content": "Você é um especialista em RH. Retorne apenas JSON válido."},
            {"role": "user", "content": prompt}
        ]
    )
    conteudo = response.choices[0].message.content

    try:
        candidato = json.loads(limpar_json(conteudo))
        skills = normalizar_scores(candidato.get("skills", []))
        candidato["skills"] = skills
        candidato["score_geral"] = calcular_score_geral(skills)
        candidato["nivel_profissional"] = calcular_nivel_profissional(candidato["score_geral"], candidato.get("anos_experiencia", 0))
        return candidato
    except Exception as e:
        print(f"Erro ao converter JSON:\n{conteudo}")
        raise e

def encontrar_melhor_candidato(descricao_vaga):
    conn = sqlite3.connect("curriculos.db")
    df_cand = pd.read_sql("SELECT nome, anos_experiencia, nivel_profissional, skills FROM candidatos", conn)
    conn.close()

    if df_cand.empty:
        return "Nenhum candidato no banco de dados."

    resumo_candidatos = df_cand.to_dict("records")
    
    prompt = f"""
    Você é um recrutador Tech Senior. 
    Abaixo está a descrição de uma vaga e uma lista de candidatos em JSON.
    Avalie os perfis e escolha O MELHOR candidato para a vaga.
    
    Escreva um parágrafo curto (máximo 4 linhas) explicando por que essa pessoa é a melhor escolha, citando as skills e a experiência dela que batem com a vaga.
    
    Descrição da Vaga:
    {descricao_vaga}
    
    Candidatos:
    {json.dumps(resumo_candidatos, ensure_ascii=False)}
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        messages=[{"role": "system", "content": "Você é um especialista em RH."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# Função Auxiliar para Gráficos Vazios bonitos e sem quebrar o layout
def gerar_grafico_vazio(titulo):
    fig = go.Figure()
    fig.update_layout(
        title=titulo,
        height=320,
        template="plotly_dark",
        xaxis={"visible": False}, # Esconde a linha com números do eixo X
        yaxis={"visible": False}, # Esconde a linha com números do eixo Y
        annotations=[{
            "text": "Aguardando currículos...",
            "xref": "paper",
            "yref": "paper",
            "showarrow": False,
            "font": {"size": 14, "color": "#777"}
        }],
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig


# ==========================================
# 4. APLICAÇÃO DASH (FRONTEND)
# ==========================================

app = Dash(__name__)

app.layout = html.Div(
    style={
        "backgroundColor": "#121212", 
        "minHeight": "100vh",
        "margin": "0",
        "padding": "30px",
        "color": "#E0E0E0",
        "fontFamily": "Inter, sans-serif"
    },
    children=[
        
        # HEADER
        html.Div([
            html.H1("Dashboard RH IA 🚀", style={"margin": "0", "color": "#FFFFFF"}),
            html.P("Análise e ranqueamento inteligente de currículos.", style={"color": "#A0A0A0"})
        ], style={"marginBottom": "30px", "textAlign": "center"}),

        # CARDS DE KPIs
        html.Div(id="kpi-cards", style={"display": "flex", "flexWrap": "wrap", "gap": "20px", "marginBottom": "30px"}),

        # ÁREA DE AÇÕES (Upload e Busca de Vaga)
        html.Div([
            # Coluna de Upload
            html.Div([
                html.H3("📥 Adicionar Currículos", style={"marginTop": "0"}),
                dcc.Upload(
                    id='upload-pdf',
                    children=html.Div(['Arraste ou ', html.A('Selecione os PDFs')]),
                    style={
                        'width': '100%', 'height': '60px', 'lineHeight': '60px',
                        'borderWidth': '2px', 'borderStyle': 'dashed',
                        'borderRadius': '10px', 'textAlign': 'center', 'borderColor': '#5C6BC0',
                        'cursor': 'pointer', 'backgroundColor': '#1E1E1E'
                    },
                    multiple=True 
                ),
                html.Div(id="upload-status", style={"marginTop": "15px", "textAlign": "center"})
            ], style={"flex": "1", "minWidth": "300px", "padding": "20px", "backgroundColor": "#1E1E1E", "borderRadius": "15px", "boxShadow": "0 4px 6px rgba(0,0,0,0.3)"}),

            # Coluna de Match de Vaga
            html.Div([
                html.H3("🎯 Encontrar Melhor Match", style={"marginTop": "0"}),
                dcc.Textarea(
                    id='input-vaga',
                    placeholder="Cole a descrição da vaga aqui (ex: Preciso de um dev React Pleno com Docker...)",
                    style={"width": "100%", "height": "80px", "borderRadius": "8px", "padding": "10px", "backgroundColor": "#2A2A2A", "color": "white", "border": "none"}
                ),
                html.Button(
                    "Analisar com IA", 
                    id="btn-match", 
                    n_clicks=0,
                    style={"marginTop": "10px", "backgroundColor": "#5C6BC0", "color": "white", "border": "none", "padding": "10px 20px", "borderRadius": "8px", "cursor": "pointer", "fontWeight": "bold", "width": "100%"}
                ),
            ], style={"flex": "1", "minWidth": "300px", "padding": "20px", "backgroundColor": "#1E1E1E", "borderRadius": "15px", "boxShadow": "0 4px 6px rgba(0,0,0,0.3)"})

        ], style={"display": "flex", "flexWrap": "wrap", "gap": "20px", "marginBottom": "30px"}),

        # TEXTINHO EXPLICATIVO DA LLM
        html.Div(
            id="resultado-match",
            style={"padding": "20px", "backgroundColor": "#2E7D32", "borderRadius": "10px", "marginBottom": "30px", "display": "none", "color": "white"}
        ),

        # GRÁFICOS E FILTROS 
        # (Coloquei uma restrição de overflow aqui para não deixar nada vazar da div)
        html.Div([
            html.Div([dcc.Graph(id="grafico_niveis")], style={"flex": "1", "minWidth": "300px", "backgroundColor": "#1E1E1E", "borderRadius": "15px", "padding": "15px", "boxShadow": "0 4px 6px rgba(0,0,0,0.3)", "overflow": "hidden"}),
            html.Div([dcc.Graph(id="grafico_skills")], style={"flex": "2", "minWidth": "300px", "backgroundColor": "#1E1E1E", "borderRadius": "15px", "padding": "15px", "boxShadow": "0 4px 6px rgba(0,0,0,0.3)", "overflow": "hidden"})
        ], style={"display": "flex", "flexWrap": "wrap", "gap": "20px", "marginBottom": "30px"}),

        # Botão Limpar Filtros isolado em sua própria Div
        html.Div([
            html.Button("Limpar Filtros", id="btn_limpar", n_clicks=0, style={"backgroundColor":"#dc3545", "color":"white", "border":"none", "padding":"10px 20px", "borderRadius":"8px", "cursor":"pointer"})
        ], style={"marginBottom": "20px"}),

        # TABELA
        html.Div([
            dash_table.DataTable(
                id="tabela_candidatos",
                page_size=10,
                sort_action="native",
                style_table={"overflowX": "auto", "borderRadius": "10px"},
                style_header={"backgroundColor": "#333", "color": "white", "fontWeight": "bold", "border": "none"},
                style_cell={"backgroundColor": "#222", "color": "#E0E0E0", "textAlign": "left", "padding": "15px", "borderBottom": "1px solid #444"}
            )
        ], style={"backgroundColor": "#1E1E1E", "padding": "20px", "borderRadius": "15px", "boxShadow": "0 4px 6px rgba(0,0,0,0.3)"})
    ]
)

# ==========================================
# 5. CALLBACKS (INTERATIVIDADE)
# ==========================================

# Match da Vaga (LLM)
@app.callback(
    [Output("resultado-match", "children"), Output("resultado-match", "style")],
    Input("btn-match", "n_clicks"),
    State("input-vaga", "value"),
    prevent_initial_call=True
)
def realizar_match(n_clicks, texto_vaga):
    if not texto_vaga:
        return "Por favor, insira a descrição da vaga.", {"padding": "20px", "backgroundColor": "#D32F2F", "borderRadius": "10px", "marginBottom": "30px", "color": "white", "display": "block"}
    
    explicacao = encontrar_melhor_candidato(texto_vaga)
    
    style_sucesso = {"padding": "20px", "backgroundColor": "#155724", "border": "1px solid #c3e6cb", "borderRadius": "10px", "marginBottom": "30px", "color": "#d4edda", "display": "block"}
    return html.Div([html.H4("🏆 Melhor Escolha da IA:", style={"marginTop": "0"}), html.P(explicacao)]), style_sucesso


# Upload Lote, Processamento e Atualização Geral da Interface
@app.callback(
    [Output("tabela_candidatos", "data"), Output("tabela_candidatos", "columns"),
     Output("grafico_niveis", "figure"), Output("grafico_skills", "figure"),
     Output("kpi-cards", "children"), Output("upload-status", "children")],
    [Input("upload-pdf", "contents"), Input("btn_limpar", "n_clicks"), Input("grafico_niveis", "clickData"), Input("grafico_skills", "clickData")],
    [State("upload-pdf", "filename"), State("upload-status", "children")] 
)
def atualizar_dashboard(conteudos_pdf, n_clicks_limpar, click_nivel, click_skill, nomes_arquivos, status_atual):
    ctx = callback_context
    trigger = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    # Preserva o status do upload ao limpar filtros normais
    msg_upload = status_atual 
    
    if trigger == "upload-pdf" and conteudos_pdf is not None:
        if not isinstance(conteudos_pdf, list):
            conteudos_pdf = [conteudos_pdf]
            nomes_arquivos = [nomes_arquivos]
            
        teve_erro = False
        for conteudo, nome in zip(conteudos_pdf, nomes_arquivos):
            try:
                content_type, content_string = conteudo.split(',')
                decoded = base64.b64decode(content_string)
                caminho_temp = f"temp_{nome}"
                
                with open(caminho_temp, "wb") as f:
                    f.write(decoded)
                
                novo_candidato = analisar_curriculo(caminho_temp)
                texto_bruto = extrair_texto_pdf(caminho_temp)
                salvar_no_banco(novo_candidato, texto_bruto)
                
                os.remove(caminho_temp) 
            except Exception as e:
                teve_erro = True
                print(f"Erro em {nome}: {str(e)}")
                
        if teve_erro:
            msg_upload = html.Div("⚠️ Alguns arquivos apresentaram erro.", style={"color": "#FFC107", "fontWeight": "bold", "fontSize": "16px"})
        else:
            msg_upload = html.Div("✅ Arquivos processados com sucesso!", style={"color": "#4CAF50", "fontWeight": "bold", "fontSize": "16px"})

    # Buscar dados do SQLite
    conn = sqlite3.connect("curriculos.db")
    df = pd.read_sql("SELECT * FROM candidatos", conn)
    conn.close()

    # Filtros ativos nos gráficos
    df_filtrado = df.copy()
    if trigger != "btn_limpar":
        if click_nivel:
            nivel = click_nivel["points"][0]["label"]
            df_filtrado = df_filtrado[df_filtrado["nivel_profissional"] == nivel]
        if click_skill:
            skill = click_skill["points"][0]["x"]
            def check_skill(row_skills):
                try:
                    s_list = json.loads(row_skills)
                    return any(s.get("nome") == skill for s in s_list)
                except: return False
            df_filtrado = df_filtrado[df_filtrado["skills"].apply(check_skill)]

    # Estruturar Tabela
    colunas_tabela = ["nome", "email", "cidade", "anos_experiencia", "score_geral", "nivel_profissional"]
    dados_tabela = df_filtrado[colunas_tabela].to_dict("records") if not df_filtrado.empty else []
    cols = [{"name": c.replace("_", " ").title(), "id": c} for c in colunas_tabela]

    # Estruturar KPIs
    total = len(df)
    score_medio = round(df["score_geral"].mean(), 1) if not df.empty else 0
    estilo_kpi = {"flex": "1", "minWidth": "200px", "padding": "20px", "background": "#1E1E1E", "borderRadius": "15px", "textAlign": "center", "boxShadow": "0 4px 6px rgba(0,0,0,0.3)", "borderTop": "4px solid #5C6BC0"}
    kpis = [
        html.Div([html.H2(total, style={"margin": "0"}), html.P("Total de Currículos")], style=estilo_kpi),
        html.Div([html.H2(score_medio, style={"margin": "0"}), html.P("Score Médio Geral")], style=estilo_kpi),
    ]

    # Estruturar Gráficos
    if df.empty:
        # Chama a nossa função para gerar os gráficos limpos
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

    return dados_tabela, cols, fig_niveis, fig_skills, kpis, msg_upload

if __name__ == "__main__":
    app.run(debug=True)