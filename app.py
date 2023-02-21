import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import matplotlib.pyplot as plt
from pyvis.network import Network

from parm import model
from parm import util as pu
from parm import units
from parm import default_param_values, parameter_masks
from parm import data
from pyvipr.util_networkx import to_networkx
from pyvipr.pysb_viz.static_viz import PysbStaticViz
import util


st.title("PARM App")
st.write(model.name)
exp_data = data.training_data()
param_names = [param.name for param in model.parameters]
param_values = np.copy(default_param_values)
two_at_mask = parameter_masks["TAT_0"]
v_extra = model.parameters['Vextra'].value
# Cell volume
v_cell = model.parameters['Vcyto'].value

# Protein-protein association rate constants -- including 2AT+PAR2 also
ppf_params = ['kf_TAT_PAR2_A_bind_Gaq', 'kr_heterotrimer_dissociation',
              'kf_reversible_heterotrimer_reassociation', 'kf_rgs_bind_gaq',
              'kf_PLC_bind_Gaq', 'kr_gaq_dissociation', 'kf_PAR2_bind_TAT']
# Other forward binding rate constants -- we'll assume these are constrained to the
# same ranges as the protein-protein interactions.
bkf_params = ['k_gdp_bind', 'k_gtp_bind', 'kf_PLC_bind_PIP2']

kf = 1e6 # (M*s)^-1
kf /= (units.M_to_molec_per_pL * v_cell) # convert to 1/(s*molec/cell)
for param_name in ppf_params + bkf_params:
    mask = [param.name==param_name for param in model.parameters]
    param_values[mask] = kf

# Mask for the PAR2 conformational change reverse (inactivation) rate.
kinactpar2_mask = [param.name=='k_inactivate_PAR2' for param in model.parameters]
# Fix the rate:
param_values[kinactpar2_mask] = 1e-2 # 1 per 100 s
n_param = len(model.parameters)
n_rate_param = 0
n_forward_rate_param = 0
n_reverse_rate_param = 0
for rule in model.rules:
    if rule.rate_forward:
        n_rate_param += 1
        n_forward_rate_param += 1
    if rule.rate_reverse:
        n_rate_param += 1
        n_reverse_rate_param += 1
st.write("Number of parameters: {}".format(n_param))
st.write("Number of kinetic rate parameters: {}".format(n_rate_param))
st.write("Number of forward kinetic rate parameters: {}".format(n_forward_rate_param))
st.write("Number of reverse kinetic rate parameters: {}".format(n_reverse_rate_param))

comp_write = st.selectbox("View componenets: ", options=['Observables', 'Parameters', 'Expressions'])
if comp_write is not None:
    if comp_write == 'Parameters':
        st.write(model.parameters)
    elif comp_write == 'Observables':
        st.write(model.observables)
    elif comp_write == 'Expressions':
        st.write(model.expressions)        
#st.write(model.rules)


with st.sidebar:

    show_network = st.checkbox("Show interactive species network")

    select_params = st.multiselect("Model parameters", param_names)

    for selected in select_params:
        st.number_input(selected, value=model.parameters[selected].value)
    
    tat_conc = st.radio("2AT concentration (nM)", [0, 10, 31.6, 100, 316, 1000, 3160, 100000])
    input_params = st.text_area("Set model parameters")
    input_params_idx = st.text_area("Set model parameters indices")

if show_network:
    model_viz = PysbStaticViz(model, generate_eqs=True)
    graph_json = model_viz.sp_view()
    #graph_nx = to_networkx(graph_json)
    graph_nx = util.to_networkx(graph_json)

    graph_pyvis = Network("500px", "500px",notebook=True,heading='PARM species graph')
    graph_pyvis.from_nx(graph_nx)
    #graph_pyvis.show_buttons(filter_=['physics'])
    graph_pyvis.show('parm-sp-view.html')

    graph_html = open('parm-sp-view.html', 'r', encoding='utf-8')
    graph_source = graph_html.read()
    components.html(graph_source, height = 550, width=550)
st.write(input_params)
st.write(input_params_idx)
input_params = np.fromstring(input_params, dtype=float, sep=',')
#st.write(input_params)
input_params_idx = np.fromstring(input_params_idx, dtype=int, sep=',')
#st.write(input_params_idx)
st.write(input_params)
st.write(input_params_idx)

for i, idx in enumerate(input_params_idx):
    line = "{} {} {}".format(idx, model.parameters[int(idx)].name, 10.**input_params[i])
    st.write(line)
    
if st.button('Run model'):
    
    times = np.linspace(0, 250, 1000)
    time_pre_equil = np.linspace(0,1000, 2000, endpoint=True)
    param_values[input_params_idx] = 10**input_params
    param_values = pu.calcium_homeostasis_reverse_rate_coupling(param_values)
    
    param_values, initials = pu.pre_equilibrate(
                             model,
                             time_pre_equil,
                             param_values=param_values,
                             tolerance=1000)
    param_values[two_at_mask] = tat_conc * units.nM_to_molec_per_pL * v_extra
    initials[0] = tat_conc * units.nM_to_molec_per_pL * v_extra                       
    model_out = pu.run_model(model, times,
                             param_values=param_values,
                             initials=initials)
    #st.write(model_out)                         
    fig, ax = plt.subplots()
    if (tat_conc > 0) and (tat_conc < 10000):
        sig_name = str(tat_conc) + "_sig"
        err_name = str(tat_conc) + "_err"
        # Get experimental measurement and std. dev.
        y_exp = exp_data[sig_name].values
        sigma_exp = exp_data[err_name].values
        ax.errorbar(exp_data['Time'], y_exp, yerr=sigma_exp, marker='s')
    ax.plot(times, model_out['FRET'])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('FRET ratio')  

    st.pyplot(fig)

    fig_2, ax_2 = plt.subplots()
    ax_2.plot(times, model_out['erCa_uM'])
    ax_2.set_xlabel('Time (s)')
    ax_2.set_ylabel('[Ca2+]_ER (uM)')
    st.pyplot(fig_2)

    fig_3, ax_3 = plt.subplots()
    ax_3.plot(times, model_out['cytoCa_nM'])
    ax_3.set_xlabel('Time (s)')
    ax_3.set_ylabel('[Ca2+]_Cyto (nM)')   
    st.pyplot(fig_3)

    fig_4, ax_4 = plt.subplots()
    ax_4.plot(times, model_out['par2_occupancy'])
    ax_4.set_xlabel('Time (s)')
    ax_4.set_ylabel('PAR2 occupancy')   
    st.pyplot(fig_4)
