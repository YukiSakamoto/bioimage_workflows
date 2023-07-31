import pathlib
import typing
import re

from typing import Tuple
PathLike = typing.Union[str, pathlib.Path]


def generation1(inputs: Tuple[PathLike, ...], output: PathLike, params: dict) -> Tuple[str, dict]:
    assert len(inputs) == 0

    seed = params['seed']
    interval = params['interval']
    num_samples = params["num_samples"]
    num_frames = params["num_frames"]
    exposure_time = params['exposure_time']
    Nm = params['Nm']
    Dm = params['Dm']
    transmat = params['transmat']

    assert num_samples < 1000
    assert num_frames < 1000

    artifacts = output

    #XXX: HERE

    import numpy
    rng = numpy.random.RandomState(seed)

    import scopyon

    config = scopyon.DefaultConfiguration()
    # config.environ.processes = 10
    config.default.effects.photo_bleaching.switch = False
    config.default.detector.exposure_time = exposure_time
    pixel_length = config.default.detector.pixel_length / config.default.magnification
    L_2 = config.default.detector.image_size[0] * pixel_length * 0.5

    timepoints = numpy.linspace(0, interval * num_frames, num_frames + 1)
    ndim = 2

    config.save(artifacts / 'config.yaml')

    for i in range(num_samples):
        samples = scopyon.sample(timepoints, N=Nm, lower=-L_2, upper=+L_2, ndim=ndim, D=Dm, transmat=transmat, rng=rng)
        inputs_data = [(t, numpy.hstack((points[:, : ndim], points[:, [ndim + 1]], numpy.ones((points.shape[0], 1), dtype=numpy.float64)))) for t, points in zip(timepoints, samples)]
        ret = list(scopyon.generate_images(inputs_data, num_frames=num_frames, config=config, rng=rng, full_output=True))

        inputs_data_ = []
        for t, data in inputs_data:
            inputs_data_.extend(([t] + list(row) for row in data))
        inputs_data_ = numpy.array(inputs_data_)
        numpy.save(artifacts / f"inputs{i:03d}.npy", inputs_data_)

        numpy.save(artifacts / f"images{i:03d}.npy", numpy.array([img.as_array() for img, infodict in ret]))
        for j, (img, _) in enumerate(ret):
            img.save(artifacts / f"image{i:03d}_{j:03d}.png")

        true_data = []
        for t, (_, infodict) in zip(timepoints, ret):
            true_data.extend([t, key] + list(value) for key, value in infodict['true_data'].items())
        true_data = numpy.array(true_data)
        numpy.save(artifacts / f"true_data{i:03d}.npy", true_data)

    return artifacts.absolute().as_uri(), {}

def analysis1(inputs: Tuple[PathLike, ...], output: PathLike, params: dict) -> Tuple[str, dict]:
    assert len(inputs) == 1

    min_sigma = params["min_sigma"]
    max_sigma = params["max_sigma"]
    threshold = params["threshold"]
    overlap = params["overlap"]
    print(f"DEBUG info min_sigma={min_sigma}, max_sigma={max_sigma}, threshold={threshold}, overlap={overlap}")
    generation_artifacts = inputs[0]
    artifacts = output

    #XXX: HERE

    import numpy

    import scopyon

    import warnings
    warnings.simplefilter('ignore', RuntimeWarning)

    num_spots = 0
    for image_npy_path in sorted(generation_artifacts.glob('images*.npy')):
        mobj = re.match('images(\d+).npy', image_npy_path.name)
        assert mobj is not None
        i = int(mobj.group(1))
        imgs = [scopyon.Image(data) for data in numpy.load(image_npy_path)]
        spots = [
            scopyon.analysis.spot_detection(
                img.as_array(),
                min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold, overlap=overlap)
            for img in imgs]

        timepoints = numpy.arange(0, len(spots), dtype=numpy.float64)

        all_spots = []
        for t, data in zip(timepoints, spots):
            all_spots.extend(([t] + list(row) for row in data))
        all_spots = numpy.array(all_spots)
        numpy.save(artifacts / f"spots{i:03d}.npy", all_spots)

        r = 6
        for j, (img, spots_) in enumerate(zip(imgs, spots)):
            shapes = [dict(x=spot[0], y=spot[1], sigma=r, color='red')
                      for spot in spots_]
            img.save(artifacts / f"spots{i:03d}_{j:03d}.png", shapes=shapes)

        num_spots += len(all_spots)
        # print("{} spots are detected in {} frames.".format(num_spots, len(imgs)))

    metrics = {"num_spots": num_spots}  #XXX: optional
    return artifacts.absolute().as_uri(), metrics

def evaluation1(inputs: Tuple[PathLike, ...], output: PathLike, params: dict) -> Tuple[str, dict]:
    #assert len(inputs) == 2
    print(inputs)
    print(params)

    max_distance = params["max_distance"]

    # generation_artifacts requires generation1's output 
    generation_artifacts = inputs[0]
    print(f"DEBUG info generation_artifacts={generation_artifacts}")
    # analysis_artifacts requires analysis1's output
    analysis_artifacts = inputs[1]
    print(f"DEBUG info analysis_artifacts={analysis_artifacts}")
    artifacts = output

    #XXX: HERE

    import numpy

    rates = numpy.zeros(4, dtype=int)

    import scopyon

    closest = []
    for true_npy_path in sorted(generation_artifacts.glob('true_data*.npy')):
        mobj = re.match('true_data(\d+).npy', true_npy_path.name)
        assert mobj is not None
        i = int(mobj.group(1))

        true_data_ = numpy.load(true_npy_path)
        t = true_data_[0, 0]
        true_data = [[true_data_[0, 1: ]]]
        for row in true_data_[1: ]:
            if row[0] == t:
                true_data[-1].append(row[1: ])
            else:
                t = row[0]
                true_data[-1] = numpy.asarray(true_data[-1])
                true_data.append([row[1: ]])
        else:
            true_data[-1] = numpy.asarray(true_data[-1])

        spots_ = numpy.load(analysis_artifacts / f"spots{i:03d}.npy")
        t = spots_[0, 0]
        spots = [[spots_[0, 1: ]]]
        for row in spots_[1: ]:
            if row[0] == t:
                spots[-1].append(row[1: ])
            else:
                t = row[0]
                spots[-1] = numpy.asarray(spots[-1])
                spots.append([row[1: ]])
        else:
            spots[-1] = numpy.asarray(spots[-1])

        for true_data_, spots_ in zip(true_data, spots):
            data = true_data_[:, 3: 5]
            for spot in spots_:
                distance = data - spot[0: 2]
                idx = (distance ** 2).sum(axis=1).argmin()
                closest.append(distance[idx])

                distance = numpy.sqrt(distance[idx] ** 2).sum()
                if distance < max_distance:
                    rates[0] += 1
                else:
                    rates[1] += 1

            for spot in data:
                distance = spots_[:, : 2] - spot
                distance = (distance ** 2).sum(axis=1)
                idx = distance.argmin()
                distance = numpy.sqrt(distance[idx])
                if distance < max_distance:
                    rates[2] += 1
                else:
                    rates[3] += 1

        imgs = [scopyon.Image(data) for data in numpy.load(generation_artifacts / f'images{i:03d}.npy')]

        r = 6
        for j, (img, spots_, true_data_) in enumerate(zip(imgs, spots, true_data)):
            shapes = [dict(x=spot[0], y=spot[1], sigma=r, color='green')
                       for spot in true_data_[:, 3: 5]]
            shapes += [dict(x=spot[0], y=spot[1], sigma=r, color='red')
                      for spot in spots_]
            img.save(artifacts / f"spots{i:03d}_{j:03d}.png", shapes=shapes)

    closest = numpy.asarray(closest).T

    x_mean, y_mean = numpy.average(closest[0]), numpy.average(closest[1])
    x_std, y_std = numpy.std(closest[0]), numpy.std(closest[1])
    # print(f"Average along x-axis = {x_mean:+.5f} pixels, std = {x_std}")
    # print(f"Average along y-axis = {y_mean:+.5f} pixels, std = {y_std}")

    import plotly.express as px
    w = h = 1
    H, xedges, yedges = numpy.histogram2d(x=closest[0], y=closest[1], bins=41, range=[[-w, +w], [-h, +h]])
    fig = px.imshow(H, x=(xedges[: -1]+xedges[1: ])*0.5, y=(yedges[: -1]+yedges[1: ])*0.5)
    # fig.show()
    fig.write_image(str(artifacts / "heatmap1.png"))

    r = rates[: 2].sum() / rates[2: ].sum()
    miss_count = rates[1] / rates[: 2].sum()
    missing = rates[3] / rates[2: ].sum()
    # print(f"The ratio between detected and expected = {r:.3f}")
    # print(f"The fraction of miss counted spots = {miss_count:.3f}")
    # print(f"The fraction of spots not detected = {missing:.3f}")

    metrics = {
        'x_mean': x_mean, "x_std": x_std, "y_mean": y_mean, 'y_std': y_std,
        'r': r, 'miss_count': miss_count, 'missing': missing,
        }
    return artifacts.absolute().as_uri(), metrics

def analysis2(inputs: Tuple[PathLike, ...], output: PathLike, params: dict) -> Tuple[str, dict]:
    assert len(inputs) == 1

    interval = params['interval']
    cutoff_distance = params["cutoff_distance"]

    generation_artifacts = inputs[0]
    artifacts = output

    # #XXX: HERE

    import scopyon
    config = scopyon.Configuration(filename=generation_artifacts / "config.yaml")
    pixel_length = config.default.detector.pixel_length / config.default.magnification

    import numpy

    def trace_spots(spots, cutoff_distance=numpy.inf, ndim=2):
        observation_vec = []
        lengths = []
        for i in range(len(spots[0])):
            iprev = i
            for j in range(1, len(spots)):
                displacements = numpy.power(spots[j][:, : ndim] - spots[j - 1][iprev, : ndim], 2).sum(axis=1)
                inext = displacements.argmin()
                displacement = numpy.sqrt(displacements[inext])
                if displacement > cutoff_distance:
                    if j > 1:
                        lengths.append(j - 1)
                    break
                intensity = spots[j - 1][iprev, ndim]
                observation_vec.append([displacement, intensity])
                iprev = inext
            else:
                lengths.append(len(spots) - 1)
        return observation_vec, lengths

    observation_vec = []
    lengths = []
    ndim = 2
    for spots_npy_path in sorted(artifacts.glob('spots*.npy')):  #XXX: Expect to be in the artifacts
        mobj = re.match('spots(\d+).npy', spots_npy_path.name)
        assert mobj is not None
        i = int(mobj.group(1))

        spots_ = numpy.load(spots_npy_path)
        t = spots_[0, 0]
        spots = [[spots_[0, 1: ]]]
        for row in spots_[1: ]:
            if row[0] == t:
                spots[-1].append(row[1: ])
            else:
                t = row[0]
                spots[-1] = numpy.asarray(spots[-1])
                spots.append([row[1: ]])
        else:
            spots[-1] = numpy.asarray(spots[-1])
        # print(spots)

        observation_vec_, lengths_ = trace_spots(spots, cutoff_distance=cutoff_distance, ndim=ndim)
        observation_vec.extend(observation_vec_)
        lengths.extend(lengths_)
    observation_vec = numpy.array(observation_vec)

    print(lengths)

    print(len(lengths), sum(lengths))

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2, subplot_titles=['Square Displacement', 'Intensity'])
    fig.add_trace(go.Histogram(x=observation_vec[:, 0], nbinsx=30, histnorm='probability'), row=1, col=1)
    fig.add_trace(go.Histogram(x=observation_vec[:, 1], nbinsx=30, histnorm='probability'), row=1, col=2)
    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.75, showlegend=False)
    # fig.show()
    fig.write_image(str(artifacts / "histogram1.png"))

    from scopyon.analysis import PTHMM

    seed = 32
    rng = numpy.random.RandomState(seed)
    model = PTHMM(n_diffusivities=3, n_oligomers=1, n_iter=100, random_state=rng)
    model.fit(observation_vec, lengths)
    
    print("diffusivities=\n", model.diffusivities_)
    print("D=\n", pixel_length ** 2 * model.diffusivities_ / interval / 1e-12)

    print("intensity_means=", model.intensity_means_)
    print("intensity_vars=", model.intensity_vars_)

    print("startprob=\n", model.startprob_)

    P = model.transmat_
    k = -numpy.log(1 - P) / interval
    k.ravel()[:: k.shape[0] + 1] = 0.0
    print("transmat=\n", model.transmat_)
    print("state_transition_matrix=\n", k)

    expected_vec = numpy.zeros((sum(lengths), 2), dtype=observation_vec.dtype)
    for i in range(len(lengths)):
        X_, Z_ = model.sample(lengths[i])
        expected_vec[sum(lengths[: i]): sum(lengths[: i + 1])] = X_

    fig = make_subplots(rows=1, cols=2, subplot_titles=['Square Displacement', 'Intensity'])
    fig.add_trace(go.Histogram(x=observation_vec[:, 0], nbinsx=30, histnorm='probability density'), row=1, col=1)
    fig.add_trace(go.Histogram(x=expected_vec[:, 0], nbinsx=30, histnorm='probability density'), row=1, col=1)
    fig.add_trace(go.Histogram(x=observation_vec[:, 1], nbinsx=30, histnorm='probability density'), row=1, col=2)
    fig.add_trace(go.Histogram(x=expected_vec[:, 1], nbinsx=30, histnorm='probability density'), row=1, col=2)
    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.75, showlegend=False)
    # fig.show()
    fig.write_image(str(artifacts / "histogram2.png"))

    #XXX: THERE
    return {"artifacts": artifacts.absolute().as_uri()}

    metrics = {'observation_count': len(lengths), 'observation_length': sum(lengths)}
    return artifacts.absolute().as_uri(), metrics
