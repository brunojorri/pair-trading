# tickers.py
import json
import os

def carregar_tickers():
    try:
        with open("meus_tickers.json", "r", encoding="utf-8") as file:
            data = json.load(file)
            tickers = data.get("tickers", [])
            if isinstance(tickers, list):
                return tickers
            else:
                print("⚠️ Formato inválido: 'tickers' deve ser uma lista.")
                return []
    except FileNotFoundError:
        print("❌ Arquivo meus_tickers.json não encontrado.")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ Erro ao ler JSON: {e}")
        return []
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        return []