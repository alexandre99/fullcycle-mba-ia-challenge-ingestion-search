from search import search_prompt
from factory import get_llm

def main():
    llm = get_llm()
    print(f"Using LLM: {type(llm).__name__}")
    
    # chain = search_prompt()
    # ...
    pass

if __name__ == "__main__":
    main()