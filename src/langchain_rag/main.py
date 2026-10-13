import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


def main() -> None:
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is missing. Add it to you .env file.")

    model_name = os.getenv("GEMINI_MODEL")

    model = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
        max_retries=2,
    )

    response = model.invoke(
        "where the surendranagar is situated?"
        "and what is it famous for? in plain text and withoud any markdowns. in 20 words"
    )

    print("GEMINI_RESPONSE:")
    print(response.text)


if __name__ == "__main__":
    main()
