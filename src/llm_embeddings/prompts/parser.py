import polars as pl

YEAR_TOKEN = "<year>"
PRESIDENT_TOKEN = "<name>"
PARTY_TOKEN = "<party>"


class TemplateParser:
    def __init__(
        self, year_col: str, president_col: str, party_col: str, template_col: str
    ):
        self.year_col = year_col
        self.president_col = president_col
        self.party_col = party_col
        self.template_col = template_col

    def parse(self, prompt_dataset):
        return prompt_dataset.with_columns(
            pl.col(self.template_col)
            .str.replace(YEAR_TOKEN, pl.col(self.year_col))
            .str.replace(PRESIDENT_TOKEN, pl.col(self.president_col))
            .str.replace(PARTY_TOKEN, pl.col(self.party_col))
            .alias("prompt")
        )
