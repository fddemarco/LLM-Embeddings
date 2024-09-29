import polars as pl
import timeness.preprocessing.constants as constants


def load_sotu_data(party, model):
    embeddings = pl.read_parquet(constants.SOTU_PATH % model)
    return filter_by_party(embeddings, party)


def filter_by_party(embeddings, party):
    if party != constants.ALL_PARTY:
        return embeddings.filter(pl.col(constants.PARTY_COL) == party).select(
            pl.exclude(constants.SPEECH_COLS)
        )
    return embeddings.select(pl.exclude(constants.SPEECH_COLS))
