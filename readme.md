# Projeto ETL Pokémon

Este projeto implementa um **pipeline ETL (Extract, Transform, Load)** para consumir dados de uma API de Pokémons.  
O processo coleta informações de Pokémons, detalhes individuais e histórico de combates, salva os dados brutos em JSON e gera versões processadas em CSV.
---

## 📂 Estrutura de pastas

```
.
├── app/ # (se usar interface Streamlit)
│   └── app.py    # streamlit run app/app.py       
├── data/    
│   ├── raw/              # Dados brutos em JSON
│   └── processed/        # Dados tratados em CSV
├── main.py               # Ponto de entrada (opcional)
├── src/
│   └── api.py            # Cliente da API + funções ETL
├── .env                  # Variáveis de ambiente (não versionar)
└── README.md             # Este arquivo
```

---

## ⚙️ Configuração

### 1. Clonar o repositório

```bash
git clone https://github.com/PhillypeAquino/Raizen.git
```

### 2. Criar ambiente virtual e instalar dependências

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

Se não houver `requirements.txt`, instale manualmente:

```bash
pip install pandas httpx backoff python-dotenv
```

### 3. Criar arquivo `.env`

Crie o arquivo `.env` na raiz do projeto com suas credenciais:

```
BASE_URL=http://seu-servidor:8000
API_USERNAME=seu_usuario
API_PASSWORD=sua_senha
```

---

## ▶️ Execução

### Rodar o ETL via `main.py`

```bash
python main.py
```

### Rodar o Streamlit (opcional)

```bash
streamlit run app/app.py
```

---

## 📊 Saídas

Após rodar o ETL, você terá:

- `data/raw/pokemons_raw.json` → lista bruta de Pokémons  
- `data/processed/pokemons.csv` → CSV com `id` e `name`  
- `data/raw/pokemon_details_raw.json` → detalhes brutos dos Pokémons  
- `data/processed/pokemon_details.csv` → detalhes tratados em CSV  
- `data/raw/combats_raw.json` → histórico bruto de combates  
- `data/processed/combats.csv` → combates processados (se existirem)

---

## 🛠️ Tecnologias usadas

- [Python 3.10+](https://www.python.org/)  
- [httpx](https://www.python-httpx.org/) → cliente HTTP  
- [backoff](https://github.com/litl/backoff) → retry com recuo exponencial  
- [pandas](https://pandas.pydata.org/) → manipulação de dados  
- [python-dotenv](https://saurabh-kumar.com/python-dotenv/) → variáveis de ambiente  
- [Streamlit](https://streamlit.io/) (opcional) → interface interativa  

---

## 📌 Observações

- O código implementa **cache de token JWT** em `.token_cache.json` para evitar logins repetidos.  
- Se a API limitar chamadas (`429 Too Many Requests`), há `sleep` automático para respeitar o servidor.  
- O projeto é facilmente adaptável para outras APIs paginadas.
- PROJETO FEITO COM AUXILIO DE IA (Chat GPT)
---

## 🚀 Autor

Projeto desenvolvido por **Phillype Freitas** ✨
