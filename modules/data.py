from pydantic import BaseModel, Field

class PartialQueryAnswer(BaseModel):
    answer: str             = Field(description="The answer to the user's question based on the database query results if it exists. Can also only contain a partial result.")
    needs: list[str] | None = Field(description="If the question cannot be answered with the current information, return a list of strings describing what information is needed to answer the question. This can include information that is not in the database, but that would be needed to formulate the correct SQL query or to interpret the results.")
