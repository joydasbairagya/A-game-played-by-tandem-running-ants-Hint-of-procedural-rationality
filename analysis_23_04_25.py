import numpy as np
import pandas as pd
from scipy.stats import chisquare
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import wilcoxon
from scipy.stats import shapiro


# Enable LaTeX rendering
plt.rcParams['text.usetex'] = True

# Colony names corresponding to file names
colony_names = ['1993', '2020', '2048', '2049', '2103', '2108']
All_time = []
All_position = []
distance_ON_bridge= 5.1
Total_distance=77
velosity_R = 6.5
velosity_T = 4.0
Not_turn_time=np.loadtxt('All_time_not_turn.txt')
mean_not_turn_time=np.mean(Not_turn_time)

lamda = np.linspace(0,5,1000)
epsilon = np.linspace(0,velosity_T/(velosity_R + velosity_T),100)
Min_lamda=np.zeros(len(epsilon))

data_rows = []

for colony in colony_names:
    file_path = f'TIME-DISTANCE_DI_{colony}_ON.txt'

    # Load data
    data_tr = np.genfromtxt(file_path, delimiter=' ', dtype=None, encoding='utf-8', names=True)

    # Extract relevant columns
    # Type_of_Leader Decision Required_time Position Velocity
    Required_time = data_tr['Required_time']
    Position = data_tr['Position']
    interaction_position = distance_ON_bridge*np.ones(len(Position)) + Position

    All_time = np.append(Required_time,All_time)
    All_position = np.append(interaction_position,All_position)
    # Add one row per time-position pair
    for time_val, pos_val in zip(Required_time, interaction_position):
        data_rows.append({
            'Colony Name': colony,
            'Decision taking time': time_val,
            'Interaction distance from ON': pos_val
        })

# Convert all data to a DataFrame
df_i = pd.DataFrame(data_rows)

# Save to CSV
df_i.to_csv('Colony_Decision_Times.csv', index=False)

# Optional: Save to Excel
df_i.to_excel('Colony_Decision_Times.xlsx', index=False)

l1_VT = All_position/velosity_T
l_l1_VR = (Total_distance/velosity_R)*(np.ones(len(All_position)))- All_position/velosity_R
l_l1_VT = (Total_distance/velosity_R)*(np.ones(len(All_position)))- All_position/velosity_R




for i in range(len(epsilon)):
    epsilon_i = epsilon[i]
    b= epsilon_i * (Total_distance/velosity_R + Total_distance/velosity_T)
    for j in range(len(lamda)):
        lambd_j = lamda[j]
        b_time = b*np.ones(len(All_time)) + lambd_j*All_time
        mean2 = np.mean(b_time - l_l1_VR)
        stat2, pval2 = wilcoxon(b_time, l_l1_VR,alternative='greater')
        if pval2 < 0.05 and mean2>0:
            Min_lamda[i] = lambd_j
            # print(epsilon_i)
            break



plt.plot(epsilon,Min_lamda)
plt.savefig("lamda_epsilon.svg")
plt.show()

# # Wilcoxon Tests
# stat1, pval1 = wilcoxon(b_time, l1_VT)
# mean1 = np.mean(b_time - l1_VT)


# # Create DataFrame for plotting
# df_box = pd.DataFrame({
#      r'$\frac{b}{\alpha} + \tau$': b_time,
#     r'$\frac{l_1}{v_T}$': l1_VT,
#     r'$\frac{l-l_1}{v_R}$': l_l1_VR
# })

# # Melt for seaborn
# df_melt = df_box.melt(var_name='Type', value_name='Time')

# # Plot
# plt.figure(figsize=(9,6))
# sns.boxplot(x='Type', y='Time', data=df_melt, palette='Set2')
# plt.title('Comparison of Predicted vs Actual Time Components', fontsize=14)
# plt.ylabel('Time (s)', fontsize=12)
# plt.xlabel('Metric Type', fontsize=12)

# # Annotate comparison results
# def annotate(text, x1, x2, y, pval):
#     plt.plot([x1, x1, x2, x2], [y, y+0.3, y+0.3, y], lw=1.5, c='black')
#     plt.text((x1+x2)*.5, y+0.35, text + f"\n(p={pval:.3f})", ha='center', va='bottom', fontsize=10)

# # Y-position for annotations
# y_max = df_melt['Time'].max()
# annotate(r'$\frac{b}{\alpha} + \tau > \frac{l_1}{v_T}$' if mean1 > 0 else r'$\frac{b}{\alpha} + \tau < \frac{l_1}{v_T}$', 0, 1, y_max + 1, pval1)
# annotate(r'$\frac{b}{\alpha} + \tau > \frac{l-l_1}{v_T}$' if mean2 > 0 else r'$\frac{b}{\alpha} + \tau > \frac{l-l_1}{v_T}$', 0, 2, y_max + 4, pval2)

# plt.savefig("comparison.svg")
# plt.tight_layout()
# plt.show()




