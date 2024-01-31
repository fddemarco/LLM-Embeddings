import polars as pl


class Schema:
    def __init__(self, text_key, group_key):
        self.text_key = text_key
        self.group_key = group_key


class Dataset:
    def __init__(self, data, schema):
        self.data = data
        self.schema = schema

    def embed_text(self, model):
        return (
            self.data.with_columns(
                pl.col(self.schema.text_key)
                .map_elements(lambda text: model.embed([text])[0])
                .alias("temp")
            )
            .with_columns(
                pl.col("temp").list.to_struct(
                    fields=[str(i) for i in range(0, model.n_dimensions)]
                )
            )
            .unnest("temp")
        )
