from search import search_prompt
from dotenv import load_dotenv

load_dotenv()   

def main():
    print("Digite 'sair' para encerrar.\n")
    
    while True:
        question = input("Faça sua pergunta: ")
        
        if question.lower() in ["sair", "exit", "quit"]:
            print("Encerrando chat...")
            break
            
        if not question.strip():
            continue
            
        try:
            response = search_prompt(question)
            print(f"\nResposta: {response}")
            print("-" * 30)
        except ValueError as e:
            print(f"\n[Erro de Configuração]: {e}")
            print("Verifique se as variáveis de ambiente necessárias estão configuradas no seu arquivo .env.")
        except Exception as e:
            error_msg = str(e).lower()
            if "api_key" in error_msg or "authentication" in error_msg:
                print(f"\n[Erro de Autenticação]: A chave de API fornecida parece ser inválida.")
            elif "connection" in error_msg or "unreachable" in error_msg:
                print(f"\n[Erro de Conexão]: Não foi possível conectar ao serviço (LLM ou Banco de Dados).")
            else:
                print(f"\n[Erro Inesperado]: Ocorreu um erro ao processar sua pergunta: {e}")
                print("Se o erro persistir, verifique logs do sistema ou configurações.")

if __name__ == "__main__":
    main()