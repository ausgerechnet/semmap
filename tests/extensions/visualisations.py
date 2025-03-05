from glob import glob

# matplotlib
import matplotlib
import matplotlib.pyplot as plt
from adjustText import adjust_text
# bokeh
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.plotting import figure, output_file, show
# plotnine
from plotnine import aes, geom_point, ggplot

from semmap.semspace import SemanticSpace
from semmap.utils import read_ccc_tables


def create_semspace():

    p_query = 'item'
    paths = glob("tests/data/ufa-sz-atomkraft/*.tsv")
    df = read_ccc_tables(paths).reset_index().dropna(subset=[p_query])
    items = list(set(df[p_query]))

    print(f"creating semantic space for {len(items)} items")
    magnitude_path = (
        "/home/ausgerechnet/corpora/embeddings/magnitude/"
        "deWikiWord2Vec.magnitude"
    )
    semspace = SemanticSpace(magnitude_path)
    semspace.generate2d(items)

    df = df.join(semspace.coordinates, how='left')

    return df


# get some sem-space
df = create_semspace()

# set visualization parameters
month = "201103"
df = df.loc[df['level_0'] == month]
df['size'] = df['log_ratio'] * 2


def test_bokeh():
    print("bokeh")
    path_out = 'tests/fig/' + month + "_bokeh.html"
    output_file(path_out)
    p = figure(title=month)
    p.scatter(x='x', y='y', size='size', source=ColumnDataSource(df))
    p.xaxis[0].axis_label = ''
    p.yaxis[0].axis_label = ''
    p.frame_height = 600
    p.frame_width = 1200
    labels = LabelSet(x='x', y='y', text='item', level='glyph',
                      source=ColumnDataSource(df), render_mode='canvas')
    p.add_layout(labels)
    show(p)


def test_matplotlib():
    print("matplotlib")
    path_out = 'tests/fig/' + month + "_matplotlib.png"
    plt.style.use("ggplot")
    matplotlib.use('Agg')
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 10)
    plt.scatter(
        df['x'],
        df['y'],
        s=0
    )
    texts = [
        plt.text(
            df['x'][i],
            df['y'][i],
            df.index[i],
            fontsize=df['size'][i],
            ha='center',
            va='center'
        ) for i in range(len(df))
    ]
    adjust_text(texts, precision=.5, avoid_self=False, avoid_points=False)
    plt.savefig(path_out, dpi=300)


def test_plotnine():
    print("plotine")
    path_out = 'tests/fig/' + month + "_plotnine.png"
    p = ggplot(df, aes(label=list(df.index))) + \
        geom_point(aes('x', 'y'))
    p.save(path_out, height=40, width=60, units="cm")
