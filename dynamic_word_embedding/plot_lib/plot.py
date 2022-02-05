from mpl_toolkits.axisartist.axislines import SubplotZero
import matplotlib.pyplot as plt
import numpy as np

# All the possibility of style: 
# [u'seaborn-darkgrid', u'seaborn-notebook', u'classic', u'seaborn-ticks',
# u'grayscale', u'bmh', u'seaborn-talk', u'dark_background', u'ggplot',
# u'fivethirtyeight', u'_classic_test', u 'seaborn-colorblind', u'seaborn-deep',
# u'seaborn-whitegrid', u'seaborn- bright', u'seaborn-poster', u'seaborn-muted',
# u'seaborn-paper', u'seaborn- white', u'seaborn-pastel', u'seaborn-dark',
# u'seaborn', u'seaborn-dark- palette']

# plt.style.use('seaborn-dark')

class Plot:
    '''A Plot class that enable thorough control over axis style. It is designed
    for demonstrating dynamic word embeddings -- x-axis has years as labels.
    '''
    def __init__(self, start_year, end_year, incre, dpi=96, figsize=(2000/96, 500/96), fig=None, pos=111):
        '''Initialization

        Arguments:
        start_year -- int, e.g., 1880
        end_year -- int, e.g., 1890
        incre -- int, increments of timesteps
        dpi -- dots per inch, argument of plt.figure()
        figsize -- argument of plt.figure()
        '''
        self.start_year = start_year
        self.end_year = end_year
        self.incre = incre

        self.get_year_span(self.start_year, self.end_year, self.incre)
        self.get_year_ticks(self.start_year, self.end_year, self.incre)

        if fig is None:
            self.dpi = dpi
            self.fig = plt.figure(figsize=figsize, dpi=dpi)
        else:
            self.dpi = fig.dpi
            self.fig = fig
        self.ax = SubplotZero(self.fig, pos)
        self.fig.add_subplot(self.ax, sharey=self.fig.get_axes()[0] if self.fig.get_axes() else None)

    def add_title(self, title_txt):
        self.ax.set_title(title_txt)

    def show(self):
        plt.show()

    def save(self, name):
        plt.savefig(name, dpi=self.dpi, bbox_inches='tight')

    def get_year_span(self, start_year, end_year, incre):
        '''Get x-axis minor tick labels from starting year, ending year, and increments.
        '''
        yearspans = []
        for year in range(start_year, end_year, incre):
            if year + incre > end_year:
                yearspan = str(year) + '-' + str(end_year)
            else:
                yearspan = str(year) + '-' + str(year + incre)
            yearspans.append(yearspan)
        self.yearspans = yearspans
        return yearspans 

    def get_year_ticks(self, start_year, end_year, incre):
        '''Get x-axis tick labels from starting year, ending year, and increments.
        '''
        self.yearticks = list(range(start_year, end_year + 1, incre))
        return self.yearticks

class PlotMeaningShift(Plot):
    def __init__(self, start_year, end_year, incre, dpi=96, figsize=(2000/96, 500/96), fig=None, pos=111):
        Plot.__init__(self, start_year, end_year, incre, dpi=dpi, figsize=figsize, fig=fig, pos=pos)
        
    def set_style(self):
        '''An axis style designed for scipt shifts_measure.py
        '''
        for direction in ["left", "right", "bottom", "top"]:
            # hides borders
            self.ax.axis[direction].set_visible(False)

        # adds arrows at the ends of each axis
        self.ax.axis['bottom'].set_axisline_style("-|>")
        # adds X-axis
        self.ax.axis['bottom'].set_visible(True)
        # self.ax.axis['left'].set_visible(True)

        yearspans =  self.get_year_span(self.start_year, self.end_year, self.incre)
        num_spans = len(yearspans)
        yearticks = self.get_year_ticks(self.start_year, self.end_year, self.incre)
        self.ax.set_xticks(range(num_spans + 1))
        self.ax.set_xticks(0.5 + np.array(range(num_spans)), minor=True)
        self.ax.set_xticklabels(yearspans, minor=True)
        self.ax.axis["bottom"].major_ticklabels.set_visible(False)
        self.ax.axis["bottom"].minor_ticks.set_visible(False)

        self.ax.set_xlim(-.25, num_spans + .25)
        self.ax.set_ylim(0, 1)

    def plot_wordset(self, year, words, offset=0, tot_group=1, color='black'):
        '''Plot a set of words with the most meaning shifts for a period of time. 
        The alignment is automatic.

        Arguments:
        year -- Starting year of the period.
        words -- List of words to be plotted.
        offset -- Index of the groups. Smaller index will be plotted in the lower 
            part of the canvas. Index starts with 0 (< tot_group)
        tot_group -- Total number of groups.
        color -- Color of words.
        '''
        
        if tot_group == 1:
            margin = 0.03
            group_pad = 0.0
        else:
            margin = 0.03
            group_pad = 0.2
        group_height = (1. - group_pad * (tot_group - 1) - margin * 2) / tot_group
        offset = offset * (group_height + group_pad) + margin

        percent = group_height / (len(words) + 1)
        x_idx = self.yearticks.index(year) + 0.5
        y_idx = 1
        words.reverse()
        for word in words:
            self.ax.text(x_idx, offset + y_idx * percent, word, 
                    horizontalalignment='center', 
                    verticalalignment='center',
                    size=11, 
                    color=color, 
                    zorder=5,
                    # weight='semibold',
                    # family='fantasy',
                    # name='Arial',
                    )
            y_idx += 1

        if offset - margin != 0:
            y_line = offset - 0.5 * group_pad
            self.ax.plot(self.ax.get_xlim(), [y_line, y_line], 
                        ls='--',
                        # lw=5,
                        color='grey',
                        zorder=5,
                        )

    def plot_legends(self):
        '''Plot legends for each group.
        '''
        props = dict(
                    # boxstyle='round', 
                    # facecolor='wheat', 
                    alpha=0.5)

        # self.ax.text(-0.6, 0.9, 'frequent words that changed the most', 
        self.ax.text(6, 0.53, 'frequent words that changed the most', 
                    # transform=self.ax.transAxes, 
                    fontsize=11,
                    weight='semibold',
                    color='grey',
                    verticalalignment='center', 
                    horizontalalignment='center', 
                    # bbox=props,
                    )

        self.ax.text(6, 0.47, 'sample words that began to be used', 
                    # transform=self.ax.transAxes, 
                    fontsize=11,
                    weight='semibold',
                    color='grey',
                    verticalalignment='center', 
                    horizontalalignment='center',
                    # bbox=props,
                    )

class PlotTrajectory(Plot):
    def __init__(self, start_year, end_year, incre, dpi=96, figsize=(900/96, 500/96), fig=None, pos=111):
        Plot.__init__(self, start_year, end_year, incre, dpi=dpi, figsize=figsize, fig=fig, pos=pos)

    def set_style(self, y_bom=-0.05, y_up=1.05, auto_xlim=False, tickgap=5, x_left_zero=100, x_right_lim_offset=0, show_leftticklabels=True):
        '''An axis style designed for scipt trajectory.py.

        Arguments:
        y_bom -- lower y-axis limit
        y_up -- upper y-axis limit
        tickgap -- distance between shown ticklabels based on `incre`: 
            `year_diff` = `tickgap` * `incre`
        '''
        # hides borders
        # self.ax.axis['right'].set_visible(False)
        # self.ax.axis['top'].set_visible(False)
        self.ax.axis["right"].major_ticks.set_visible(False)
        self.ax.axis["right"].minor_ticks.set_visible(False)
        self.ax.axis["top"].major_ticks.set_visible(False)
        self.ax.axis["top"].minor_ticks.set_visible(False)

        # adds arrows at the ends of each axis
        # self.ax.axis['bottom'].set_axisline_style("-|>")
        # self.ax.axis['bottom'].set_visible(False)
        # self.ax.axis['y=0.5'] = self.ax.new_floating_axis(nth_coord=0, value=0.5)
        # self.ax.axis['y=0.5'].set_axisline_style("-|>")

        self.yearticks = self.get_year_ticks(self.start_year, self.end_year, self.incre)
        self.ax.set_xticks(range(len(self.yearticks)))
        minor_ticks = np.array(range(0, len(self.yearticks) + 1, tickgap))
        minor_ticklables = np.array([str(i) for i in self.yearticks])
        minor_ticklables = minor_ticklables[minor_ticks]
        self.ax.set_xticks(minor_ticks, minor=True)
        self.ax.set_xticklabels(minor_ticklables, minor=True)
        self.ax.axis["bottom"].major_ticklabels.set_visible(False)
        self.ax.axis["bottom"].major_ticks.set_visible(False)

        self.ax.set_yticks(np.arange(-1, 1 + 1e-6, 0.1), minor=True)
        self.ax.axis["left"].minor_ticks.set_visible(False)
        self.ax.axis["left"].major_ticklabels.set_visible(show_leftticklabels)
        
        # set tickslabel fontsize
        self.ax.axis['bottom'].minor_ticklabels.set_fontsize(11)
        self.ax.axis['left'].major_ticklabels.set_fontsize(11)
        
        # set ticks length
        self.ax.axis['bottom'].minor_ticks.set_ticksize(6)
        self.ax.axis['left'].major_ticks.set_ticksize(6)

        # add x-axis ticklabels inside the plot
#         self.ax.axis["bottom"].set_ticklabel_direction("-")
#         self.ax.axis["bottom"].minor_ticklabels.set_rotation(180)
#         self.ax.axis["bottom"].minor_ticks.set_tick_out(False)
#         self.ax.axis["bottom"].major_ticks.set_tick_out(False)

        if not auto_xlim:
            # leave about 100pt left to zero for printing words if font size = 11
            x_left_lim = - x_left_zero / (self.fig.get_size_inches()[0] * self.fig.dpi / len(self.yearticks))
            # print(self.fig.get_size_inches()[0] * self.fig.dpi / len(self.yearticks), x_left_lim)
            x_right_lim = -x_left_lim + len(self.yearticks) - 1 + x_right_lim_offset
            self.ax.set_xlim(x_left_lim, x_right_lim)
            
        self.ax.set_ylim(y_bom, y_up)

        # plot halfline
        # y_dashline = 0.5
        # self.ax.plot(self.ax.get_xlim(), [y_dashline, y_dashline], 
        #             ls='--',
        #             lw=1,
        #             color='lightgrey',
        #             )

    def plot_wordset(self, year, words, offset, marker_offset=None, upwards=True, color='black', wsize=13, alpha=1.0, x_offset=0.0, group_height=0.6):
        '''Plot a set of nearest neighbors of a word at a timestep.

        Arguments:
        year -- Ending year of the timestep.
        words -- List of neighbors to be plotted.
        offset -- Starting position of y-axis.
        upwards -- If true, draw words upwards.
        color -- Color of words.
        '''
        if marker_offset is None:
            marker_offset = offset
        self.ax.plot(self.yearticks.index(year), marker_offset, 
                    marker='o', 
                    color='darkorange',
                    lw=20)

        margin = 0.05
        group_height = group_height
        percent = group_height / (len(words) + 2)
        if not upwards:
            offset = offset - margin - group_height
        else:
            offset = offset + margin
        
        y_idx = 1
        x_idx = self.yearticks.index(year)
        words.reverse()
        
        for word in words:
            self.ax.text(x_idx + x_offset, offset + y_idx * percent, word, 
                    horizontalalignment='center', 
                    verticalalignment='center',
                    size=wsize, 
                    color=color, 
                    alpha=alpha,
                    zorder=5,
                    # weight='semibold',
                    # family='fantasy',
                    # name='Arial',
                    )
            y_idx += 1
        self.ax.text(x_idx + x_offset, offset + y_idx * percent, str(year), 
                    horizontalalignment='center', 
                    verticalalignment='center',
                    size=wsize, 
                    color=color, 
                    weight='semibold',
                    alpha=alpha,
                    zorder=5,
                    # family='fantasy',
                    # name='Arial',
                    )


def test_shift():
    START_YEAR = 1885
    END_YEAR = 2005
    INCRE = 10

    DPI=96

    # plot()
    canvas = PlotMeaningShift(START_YEAR, END_YEAR, INCRE, DPI)
    canvas.set_style()
    canvas.plot_wordset(1885, ['a12', 'b2423', 'c4315', 'd1', 'e34523451'], offset=1, tot_group=2)
    canvas.plot_wordset(1885, ['a', 'b', 'c', 'd', 'e'], offset=0, tot_group=2, color='blue')
    canvas.plot_legends()
    # canvas.save('./test.pdf')
    canvas.show()

def test_trajectory():
    START_YEAR = 1885
    END_YEAR = 2005
    INCRE = 2

    DPI=96

    # plot()
    canvas = PlotTrajectory(START_YEAR, END_YEAR, INCRE, DPI)
    canvas.set_style()
    canvas.plot_wordset(1885, ['a12', 'b2423', 'c4315', 'd1', 'e34523451', 'fa', '234fasd', 'zxcvs'], offset=0.5, upwards=True)
    canvas.plot_wordset(1885, ['a', 'b', 'c', 'd', 'e'], offset=0.6, upwards=False, color='blue')
    canvas.add_title('TITLE')
    # canvas.plot_legends()
    # canvas.save('./test.pdf')
    canvas.show()

if __name__ == '__main__':
    # test_shift()
    test_trajectory()
    # main()

