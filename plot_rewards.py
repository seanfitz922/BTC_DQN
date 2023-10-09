import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot_profit(profits):
    # Clear the previous plot
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Agent Profit Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Profit')
    plt.plot(profits, label='Profit')
    plt.grid()
    plt.tight_layout()
    # Display the updated plot
    plt.show(block=False)
    plt.pause(.1)
