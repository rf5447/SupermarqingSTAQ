import h5py
import pandas as pd
import numpy as np


def binomial_prb_and_error(positives, total):
    z = 1
    nt = total + z ** 2
    pt = 1 / nt * (positives + z ** 2 / 2)
    unc = z / nt * (positives * (total - positives) / total + z**2 / 4) ** 0.5
    return pt, unc


def extract_state_counts(file_name, scan_var, threshold=1):
    file = h5py.File(f"../../data/{file_name}.h5")
    data = file['datasets']['histogram_context']['histogram']['raw']
    num_points = len(data)
    num_shots = len(data["0"])
    num_pmt = len(data["0"][0])
    state_data = np.zeros((num_points, 2**num_pmt))

    all_keys = [format(i, f'0{num_pmt}b')
                for i in range(2**num_pmt)]

    for i in range(num_points):
        point_data = np.array(data['{}'.format(i)])
        point_state = {}
        for j in range(len(point_data)):
            # first pmt is the least significant digit
            state_key = ''.join(
                [str(c) for c in [('1' if c > threshold else '0') for c in point_data[j]]])[::-1]
            if state_key not in point_state:
                point_state[state_key] = 1
            else:
                point_state[state_key] += 1
        for k in range(len(all_keys)):
            state_data[i][k] = point_state.get(all_keys[k], 0)

    x = np.array(file['datasets']['scan'][scan_var])
    df = pd.DataFrame({}, index=x[:num_points])

    for k in range(len(all_keys)):
        df[all_keys[k]] = state_data[:, k]

    df.sort_index(inplace=True)
    return df, num_shots


def extract_state_probability(file_name, scan_var, threshold=1, use_binomial=True):
    # file = h5py.File(f"../../../Downloads/{file_name}.h5")
    file = h5py.File(f"../../data/{file_name}.h5")
    data = file['datasets']['histogram_context']['histogram']['raw']
    num_points = len(data)
    num_shots = len(data["0"])
    num_pmt = len(data["0"][0])
    state_data = np.zeros((num_points, 2**num_pmt))
    state_err = np.zeros((num_points, 2**num_pmt))

    all_keys = [format(i, f'0{num_pmt}b')
                for i in range(2**num_pmt)]

    for i in range(num_points):
        point_data = np.array(data['{}'.format(i)])
        point_state = {}
        for j in range(len(point_data)):
            # first pmt is the least significant digit
            state_key = ''.join(
                [str(c) for c in [('1' if c > threshold else '0') for c in point_data[j]]])[::-1]
            if state_key not in point_state:
                point_state[state_key] = 1
            else:
                point_state[state_key] += 1
        for k in range(len(all_keys)):
            if use_binomial:
                state_data[i][k], state_err[i][k] = binomial_prb_and_error(
                    positives=point_state.get(all_keys[k], 0), total=num_shots)
            else:
                state_data[i][k] = point_state.get(all_keys[k], 0) / num_shots
                state_err[i][k] = 0

    x = np.array(file['datasets']['scan'][scan_var])
    df = pd.DataFrame({}, index=x[:num_points])

    for k in range(len(all_keys)):
        df[all_keys[k]] = state_data[:, k]
        df[f"{all_keys[k]}_err"] = state_err[:, k]

    df.sort_index(inplace=True)
    return df


def extract_state_parity(file_name, scan_var, threshold=1):
    file = h5py.File(f"../../data/{file_name}.h5")
    data = file['datasets']['histogram_context']['histogram']['raw']

    num_points = len(data)
    num_shots = len(data["0"])
    num_pmt = len(data["0"][0])
    parity_data = np.zeros((num_points))
    parity_err = np.zeros((num_points))
    pop_data = np.zeros(num_points)
    pop_err = np.zeros(num_points)

    all_keys = [format(i, f'0{num_pmt}b')
                for i in range(2**num_pmt)]

    for i in range(num_points):
        point_data = np.array(data['{}'.format(i)])
        point_state = {}
        for j in range(len(point_data)):
            state_key = ''.join(
                [str(c) for c in [('1' if c > threshold else '0') for c in point_data[j]]])
            if state_key not in point_state:
                point_state[state_key] = 1
            else:
                point_state[state_key] += 1
        p = 0
        for k in all_keys:
            if k.count('1') % 2 == 0:
                p += point_state.get(k, 0)
        positive_parity, positive_err = binomial_prb_and_error(
            positives=p, total=num_shots)
        parity_data[i] = 2 * positive_parity - 1
        parity_err[i] = 2 * positive_err
        pop_data[i] = positive_parity
        pop_err[i] = positive_err

    x = np.array(file['datasets']['scan'][scan_var])
    df = pd.DataFrame({}, index=x[:num_points])

    df['parity'] = parity_data
    df["parity_err"] = parity_err
    df['population'] = pop_data
    df['population_err'] = pop_err

    df.sort_index(inplace=True)
    return df


def extract_population_collapse_state(file_name, scan_var, pmts, threshold=1):
    file = h5py.File(f"../../data/{file_name}.h5")
    data = file['datasets']['histogram_context']['histogram']['raw']

    num_points = len(data)
    num_shots = len(data["0"])
    num_pmt = len(data["0"][0])
    parity_data = np.zeros((num_points))
    parity_err = np.zeros((num_points))
    pop_data = np.zeros(num_points)
    pop_err = np.zeros(num_points)

    all_keys = [format(i, f'0{num_pmt}b')
                for i in range(2**num_pmt)]
    assert len(pmts) == 2
    for i in range(num_points):
        point_data = np.array(data['{}'.format(i)])
        point_state = {}
        for j in range(len(point_data)):
            state_key = ''.join(
                [str(c) for c in [('1' if c > threshold else '0') for c in point_data[j]]])
            if state_key[pmts[0]] == '0' and state_key[pmts[1]] == '0':
                collapsed_state_key = '00'
            elif state_key[pmts[0]] == '1' and state_key[pmts[1]] == '0':
                collapsed_state_key = '10'
            elif state_key[pmts[0]] == '0' and state_key[pmts[1]] == '1':
                collapsed_state_key = '01'
            else:
                collapsed_state_key = '11'
            if collapsed_state_key not in point_state:
                point_state[collapsed_state_key] = 1
            else:
                point_state[collapsed_state_key] += 1
        p = 0
        for k in ['00', '01', '10', '11']:
            if k.count('1') % 2 == 0:
                p += point_state.get(k, 0)
        positive_parity, positive_err = binomial_prb_and_error(
            positives=p, total=num_shots)
        parity_data[i] = 2 * positive_parity - 1
        parity_err[i] = 2 * positive_err
        pop_data[i] = positive_parity
        pop_err[i] = positive_err

    x = np.array(file['datasets']['scan'][scan_var])
    df = pd.DataFrame({}, index=x[:num_points])

    df['parity'] = parity_data
    df["parity_err"] = parity_err
    df['population'] = pop_data
    df['population_err'] = pop_err

    df.sort_index(inplace=True)
    return df
