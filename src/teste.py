import os
import json
from openai import OpenAI 

# Inicializa o cliente OpenAI apontando para a infraestrutura da Groq
client = OpenAI(
    api_key=  "",
    base_url="https://api.groq.com/openai/v1" 
)

def montar_prompt_avaliacao(descricao_vaga, dados_candidato):
    """Template mestre com regras de negócio e Chain of Thought."""
    return f"""
    Você é um Tech Recruiter Sênior. Sua tarefa é avaliar o candidato abaixo para uma vaga específica e atribuir um 'Score de Aderência' de 0 a 100.

    DESCRIÇÃO DA VAGA:
    "{descricao_vaga}"

    DADOS DO CANDIDATO:
    {json.dumps(dados_candidato, ensure_ascii=False, indent=2)}

    REGRAS DE PONTUAÇÃO (Siga estritamente):
    - 90 a 100: Candidato ideal. Possui os requisitos obrigatórios e a senioridade exigida.
    - 70 a 89: Bom candidato. Faltam apenas alguns diferenciais ou a senioridade é um pouco menor, mas a stack principal está correta.
    - 40 a 69: Tem bastante experiência em programação/tecnologia, mas a stack principal difere um pouco da exigida. Não zere a nota, valorize a bagagem técnica prévia.
    - 0 a 39: Candidato de área totalmente diferente ou sem nenhuma experiência técnica aproveitável.

    FORMATO DE SAÍDA OBRIGATÓRIO (JSON):
    {{
        "raciocinio": "Breve justificativa passo a passo baseada nas regras acima.",
        "score_final": 0
    }}
    """

# --- DADOS DE TESTE ---

vaga_teste = """
Desenvolvedor Backend Pleno. 
Requisitos obrigatórios: Experiência sólida com PHP, framework Laravel, criação de APIs RESTful, banco de dados relacional (MySQL/PostgreSQL) e conhecimentos em segurança de dados/LGPD. 
Diferenciais: Experiência em montar fluxos de CI/CD (GitHub Actions) e deploy na AWS.
"""

candidatos_mock = [
    {
        "nome": "Carlos (O Falso Perfeito)",
        "anos_experiencia": 8,
        "nivel_profissional": "Senior",
        "skills": [{"nome": "Python"}, {"nome": "Django"}, {"nome": "AWS"}, {"nome": "GitHub Actions"}, {"nome": "LGPD"}],
        "nota_esperada": 35 
    },
    {
        "nome": "Mariana (Na Mosca)",
        "anos_experiencia": 4,
        "nivel_profissional": "Pleno",
        "skills": [{"nome": "PHP"}, {"nome": "Laravel"}, {"nome": "MySQL"}, {"nome": "GitHub Actions"}, {"nome": "AWS"}, {"nome": "APIs REST"}],
        "nota_esperada": 95
    },
    {
        "nome": "Roberto (O Desatualizado)",
        "anos_experiencia": 10,
        "nivel_profissional": "Senior",
        "skills": [{"nome": "PHP"}, {"nome": "CodeIgniter"}, {"nome": "jQuery"}, {"nome": "MySQL"}, {"nome": "HTML/CSS"}],
        "nota_esperada": 55
    },
    {
        "nome": "Lucas (O Júnior Promissor)",
        "anos_experiencia": 1,
        "nivel_profissional": "Junior",
        "skills": [{"nome": "PHP"}, {"nome": "Laravel"}, {"nome": "PostgreSQL"}, {"nome": "APIs RESTful"}, {"nome": "LGPD"}],
        "nota_esperada": 75
    },
    {
        "nome": "Ana (A Front Disfarçada)",
        "anos_experiencia": 3,
        "nivel_profissional": "Pleno",
        "skills": [{"nome": "JavaScript"}, {"nome": "React"}, {"nome": "Node.js"}, {"nome": "PHP"}, {"nome": "MySQL"}],
        "nota_esperada": 45
    }
]

# --- LÓGICA DE AVALIAÇÃO ---

def testar_acuracia():
    print("Iniciando teste de acurácia com JSON Mode Nativo...\n")
    erros = []

    for candidato in candidatos_mock:
        nome = candidato['nome']
        nota_esperada = candidato['nota_esperada']

        dados_para_ia = {k: v for k, v in candidato.items() if k != 'nota_esperada'}
        prompt = montar_prompt_avaliacao(vaga_teste, dados_para_ia)

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "Você é um assistente especializado em RH Tech. Responda OBRIGATORIAMENTE em formato JSON válido."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                # A MÁGICA ACONTECE AQUI:
                response_format={"type": "json_object"} 
            )
            
            # Pegamos a resposta e jogamos DIRETO no json.loads
            resposta_bruta = response.choices[0].message.content
            resultado = json.loads(resposta_bruta)
            
            nota_ia = resultado.get("score_final", 0)
            raciocinio = resultado.get("raciocinio", "Sem justificativa")
            
            erro = abs(nota_esperada - nota_ia)
            erros.append(erro)

            print(f"[{nome}] Gabarito: {nota_esperada} | IA: {nota_ia} | Erro: {erro} pontos")
            print(f"   ↳ Raciocínio: {raciocinio}\n")

        except json.JSONDecodeError:
            print(f"[{nome}] Erro: A IA não retornou um JSON válido.")
        except Exception as e:
            print(f"[{nome}] Erro ao processar a resposta da API: {e}")

    if erros:
        mae = sum(erros) / len(erros)
        print("--- ESTATÍSTICAS ---")
        print(f"Erro Médio Absoluto (MAE): {mae:.2f} pontos")
        if mae <= 10:
            print("✅ Acurácia EXCELENTE. O modelo está calibrado perfeitamente.")
        elif mae <= 20:
            print("⚠️ Acurácia ACEITÁVEL. O modelo está bom, mas pode sofrer pequenos ajustes.")
        else:
            print("❌ Acurácia RUIM. O modelo ainda precisa de calibração nas regras.")

if __name__ == "__main__":
    testar_acuracia()