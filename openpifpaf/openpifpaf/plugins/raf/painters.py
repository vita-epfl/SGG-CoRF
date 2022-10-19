import numpy as np
from graphviz import Digraph
from collections import defaultdict

import colorsys

try:
    import matplotlib
    import matplotlib.animation
    import matplotlib.collections
    import matplotlib.patches
except ImportError:
    matplotlib = None

class RelationPainter:
    def __init__(self, *, xy_scale=1.0, eval=True):
        self.xy_scale = xy_scale
        self.G = None
        if eval:
            self.G = Digraph(format='png')
        self.dict_nodes = defaultdict(lambda: defaultdict(str))
        self.node_count = 0

    def annotations(self, ax, annotations, *,
                    color=None, colors=None, texts=None, subtexts=None):
        for i, ann in reversed(list(enumerate(annotations))):
            this_color_obj = ann.category_id_obj - 1
            this_color_sub = ann.category_id_sub - 1
            this_color_rel = ann.category_id_rel - 1
            if colors is not None:
                if len(colors[i])==3:
                    this_color_obj = colors[i][2]
                    this_color_sub = colors[i][0]
                    this_color_rel = colors[i][1]
                else:
                    this_color_obj = colors[i]
                    this_color_sub = colors[i]
                    this_color_rel = colors[i]
            elif color is not None:
                if len(color)==3:
                    this_color_obj = color[2]
                    this_color_sub = color[0]
                    this_color_rel = color[1]
                else:
                    this_color_obj = color
                    this_color_sub = color
                    this_color_rel = color

            text_sub = ann.category_sub
            text_obj = ann.category_obj
            text_rel = ann.category_rel
            if texts is not None:
                text_sub = texts[i]
                text_obj = texts[i]
                text_rel = texts[i]

            subtext_sub = None
            subtext_obj = None
            subtext_rel = None
            if subtexts is not None:
                subtext_sub = subtexts[i]
                subtext_obj = subtexts[i]
                subtext_rel = subtexts[i]
            elif ann.score_obj:
                subtext_obj = '{:.0%}'.format(ann.score_obj)
                subtext_sub = '{:.0%}'.format(ann.score_sub)
                subtext_rel = '{:.0%}'.format(ann.score_rel)

            self.annotation(ax, ann, color=(this_color_sub, this_color_rel,this_color_obj), text=(text_sub, text_rel, text_obj), subtext=(subtext_sub, subtext_rel, subtext_obj))
            if self.G:
                self.draw_graph(ann, color=(this_color_sub, this_color_rel,this_color_obj), text=(text_sub, text_rel, text_obj))
        if self.G:
            self.G.render('all-images/scene-graph.gv', view=False)

    def draw_graph(self, ann, color=None, text=None):
        if color is None:
            color = (ann.category_id_sub - 1, ann.category_id_obj - 1, ann.category_id_rel - 1)

        if isinstance(color, (int, np.integer)):
            color = (color, color, color)

        if isinstance(color[0], (int, np.integer)):
            color_sub = matplotlib.cm.get_cmap('tab20')((color[0] % 20 + 0.05) / 20)
            color_sub = colorsys.rgb_to_hsv(color_sub[0], color_sub[1], color_sub[2])
        else:
            color_sub = matplotlib.colors.to_rgba(color[0])
        if isinstance(color[1], (int, np.integer)):
            color_rel = matplotlib.cm.get_cmap('tab20')((color[1] % 20 + 0.05) / 20)
            color_rel = colorsys.rgb_to_hsv(color_rel[0], color_rel[1], color_rel[2])
        else:
            color_rel = matplotlib.colors.to_rgba(color[1])
        if isinstance(color[2], (int, np.integer)):
            color_obj = matplotlib.cm.get_cmap('tab20')((color[2] % 20 + 0.05) / 20)
            color_obj = colorsys.rgb_to_hsv(color_obj[0], color_obj[1], color_obj[2])
        else:
            color_obj = matplotlib.colors.to_rgba(color[2])


        if ann.category_id_sub in self.dict_nodes and tuple(ann.bbox_sub) in self.dict_nodes[ann.category_id_sub].keys():
            node_id_sub = self.dict_nodes[ann.category_id_sub][tuple(ann.bbox_sub)]
        else:
            self.node_count = self.node_count + 1
            node_id_sub = str(self.node_count)
            self.dict_nodes[ann.category_id_sub][tuple(ann.bbox_sub)] = node_id_sub

        if ann.category_id_obj in self.dict_nodes and tuple(ann.bbox_obj) in self.dict_nodes[ann.category_id_obj].keys():
            node_id_obj = self.dict_nodes[ann.category_id_obj][tuple(ann.bbox_obj)]
        else:
            self.node_count = self.node_count + 1
            node_id_obj = str(self.node_count)
            self.dict_nodes[ann.category_id_obj][tuple(ann.bbox_obj)] = node_id_obj

        self.G.node(node_id_sub, label=text[0], color="%f, %f, %f" % (color_sub[0], color_sub[1], color_sub[2]))
        self.G.node(node_id_obj, label=text[2], color="%f, %f, %f" % (color_obj[0], color_obj[1], color_obj[2]))
        self.G.edge(node_id_sub, node_id_obj, label=text[1], fillcolor="%f, %f, %f" % (color_rel[0], color_rel[1], color_rel[2]))

    def annotation(self, ax, ann, *, color=None, text=None, subtext=None):
        if color is None:
            color = (ann.category_id_sub - 1, ann.category_id_rel - 1, ann.category_id_obj - 1)

        if isinstance(color, (int, np.integer)):
            color = (color, color, color)

        if isinstance(color[0], (int, np.integer)):
            color_sub = matplotlib.cm.get_cmap('tab20')((color[0] % 20 + 0.05) / 20)
        else:
            color_sub = color[0]
        if isinstance(color[1], (int, np.integer)):
            color_rel = matplotlib.cm.get_cmap('tab20')((color[1] % 20 + 0.05) / 20)
        else:
            color_rel = color[1]
        if isinstance(color[2], (int, np.integer)):
            color_obj = matplotlib.cm.get_cmap('tab20')((color[2] % 20 + 0.05) / 20)
        else:
            color_obj = color[2]

        if text is None:
            text = (ann.category_sub, ann.category_rel, ann.category_obj)
        elif isinstance(text, str):
            text = (text, text, text)

        if subtext is not None:
            subtext = (subtext, subtext, subtext)
        elif ann.score_obj:
            subtext = ('{:.0%}'.format(ann.score_sub), '{:.0%}'.format(ann.score_rel), '{:.0%}'.format(ann.score_obj))

        # SUBJECT
        x, y, w, h = ann.bbox_sub * self.xy_scale
        if w < 5.0:
            x -= 2.0
            w += 4.0
        if h < 5.0:
            y -= 2.0
            h += 4.0

        center_sub = (x+w/2.0, y+h/2.0)
        # draw box SUBJECT
        ax.add_patch(
            matplotlib.patches.Rectangle(
                (x, y), w, h, fill=False, color=color_sub, linewidth=1.0))

        # draw text SUBJECT
        ax.annotate(
            text[0],
            (x, y),
            fontsize=8,
            xytext=(5.0, 5.0),
            textcoords='offset points',
            color='white', bbox={'facecolor': color_sub, 'alpha': 0.5, 'linewidth': 0},
        )

        # SUBJECT
        if subtext is not None:
            ax.annotate(
                subtext[0],
                (x, y),
                fontsize=5,
                xytext=(5.0, 18.0 + 3.0),
                textcoords='offset points',
                color='white', bbox={'facecolor': color_sub, 'alpha': 0.5, 'linewidth': 0},
            )

        # OBJECT
        x, y, w, h = ann.bbox_obj * self.xy_scale
        if w < 5.0:
            x -= 2.0
            w += 4.0
        if h < 5.0:
            y -= 2.0
            h += 4.0

        center_obj= (x+w/2.0, y+h/2.0)
        # draw box OBJECT
        ax.add_patch(
            matplotlib.patches.Rectangle(
                (x, y), w, h, fill=False, color=color_obj, linewidth=1.0))

        # draw text OBJECT
        ax.annotate(
            text[2],
            (x, y),
            fontsize=8,
            xytext=(5.0, 5.0),
            textcoords='offset points',
            color='white', bbox={'facecolor': color_obj, 'alpha': 0.5, 'linewidth': 0},
        )

        # OBJECT
        if subtext is not None:
            ax.annotate(
                subtext[2],
                (x, y),
                fontsize=5,
                xytext=(5.0, 18.0 + 3.0),
                textcoords='offset points',
                color='white', bbox={'facecolor': color_obj, 'alpha': 0.5, 'linewidth': 0},
            )

        #PREDICATE
        lines, line_colors = [], []
        rel_loc = ((center_sub[0]+center_obj[0])/2.0, (center_sub[1]+center_obj[1])/2.0)
        lines.append([center_sub, rel_loc])
        lines.append([center_obj, rel_loc])
        line_colors.append(color_rel)
        line_colors.append(color_rel)
        ax.add_collection(matplotlib.collections.LineCollection(
            lines, colors=line_colors,
            capstyle='round',
        ))

        ax.annotate(
            text[1],
            rel_loc,
            fontsize=8,
            xytext=(5.0, 5.0),
            textcoords='offset points',
            color='white', bbox={'facecolor': color_rel, 'alpha': 0.5, 'linewidth': 0},
        )
