#main.py
from generate import search



if __name__ == "__main__":

    while True:
        query = input("\nUSER:: ")

        if query.lower() in {"quit", "stop", "gtfo", "close"}:
            break
        
        result = search(query)

        print(f"AI:: {result['reply']}")