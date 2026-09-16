import matplotlib.pyplot as plt

def figure_compare_SoC(df_reference, df_test, label_graph_1 = 'Reference', label_graph_2 = 'Test'):

    plt.figure(figsize=(10, 5))

    # Plotting 'soc_proxy' vs 'timesteps' for reference data
    plt.plot(df_reference['timesteps'], df_reference['soc_proxy'], label=label_graph_1)

    # Plotting 'soc_proxy' vs 'timesteps' for test data
    plt.plot(df_test['timesteps'], df_test['soc_proxy'], label=label_graph_2)

    plt.xlabel('Time')
    plt.ylabel('State of Charge (proxy)')
    plt.title('Storage State of Charge Over Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()

def figure_compare_muliple_SoCs(df_reference, list_of_test_dfs, label_graph_1, list_of_test_labels):
    
    #ensure the number of labels matches
    if len(list_of_test_dfs)!=len(list_of_test_labels):
        raise Exception('Number of labels does not match number of samples.')
    
    #assign pastel colours
    colours = plt.cm.Pastel1.colors

    plt.figure(figsize=(10, 5))
    # Plotting 'soc_proxy' vs 'timesteps' for reference data  
    
    for x in range(len(list_of_test_dfs)):
        plt.plot(list_of_test_dfs[x].index, list_of_test_dfs[x]['soc_proxy'], label=list_of_test_labels[x], color=colours[x % len(colours)], zorder=1)


    plt.plot(df_reference.index, df_reference['soc_proxy'], label=label_graph_1,color='black', zorder=101)   

    plt.xlabel('Time')
    plt.ylabel('State of Charge (proxy)')
    plt.title('Storage State of Charge Over Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()