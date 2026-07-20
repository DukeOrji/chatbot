#main.py
from pydantic import BaseModel
from generate import search




if __name__ == "__main__":

    while True:
        msg = input("\nYou: ")

        if msg.lower() in {"quit", "exit"}:
            break

        query = Query(msg=msg)

        result = search(query)

        print(f"\nBot: {result['reply']}")