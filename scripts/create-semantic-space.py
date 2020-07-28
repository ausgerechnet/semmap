from pandas import read_csv
from io import StringIO
from cutils.semspace import SemanticSpace
from argparse import ArgumentParser


def read_cqpweb_table(path_in):

    with open(path_in, "rt") as f_in:
        table_rows = list()
        for line in f_in:
            if len(line.split("\t")) > 3:
                table_rows.append(line)

    df = read_csv(StringIO("".join(table_rows)), sep="\t", index_col=1)
    df = df[['Stat 1.']]
    df.columns = ['am']
    df.index.name = 'item'
    return df


def main(path1, path2, embeddings, names=None):

    print("reading data")
    df1 = read_cqpweb_table(path1)
    df2 = read_cqpweb_table(path2)
    print("%d items in first df, %d items in second df" % (len(df1), len(df2)))
    if names is None:
        names = ["1", "2"]
    df = df1.join(df2, lsuffix="_"+names[0], rsuffix="_"+names[1], how='outer')

    print("creating semantic space for %d items" % len(df))
    semspace = SemanticSpace(embeddings)
    coordinates = semspace.generate_semspace(df.index)
    df = df.join(coordinates)

    return df


if __name__ == '__main__':

    embeddings = (
        "/home/ausgerechnet/corpora/wectors/magnitude/"
        "deWikiWord2Vec.magnitude"
    )

    parser = ArgumentParser()
    parser.add_argument("path1",
                        type=str,
                        help="first file")
    parser.add_argument("path2",
                        type=str,
                        help="second file")
    parser.add_argument("path_out",
                        type=str,
                        help="path to save result to (.tsv.gz)")
    args = parser.parse_args()
    df = main(args.path1, args.path2, embeddings)
    print("saving results")
    df.to_csv(args.path_out, sep="\t", compression="gzip")
