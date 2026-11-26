import math
import warnings
from typing import List, Optional, Union, Tuple
import torch
from torch import Tensor


# ================= 3D UTILS FOR SCALAR PROPAGATOR =====================
def pad_model_3d(model: Tensor, pad: List[int], mode: str = "replicate") -> Tensor:
    """Pad a 3D model tensor with [z0, z1, y0, y1, x0, x1] padding."""
    # model: (nz, ny, nx), pad: [z0, z1, y0, y1, x0, x1]
    return torch.nn.functional.pad(model[None, None],
                                   (pad[4], pad[5], pad[2], pad[3], pad[0], pad[1]),
                                   mode=mode)[0, 0]

def pad_locations_3d(locations: Tensor, pad: List[int]) -> Tensor:
    """Pad 3D locations by adding pad starts [z0, y0, x0]."""
    pad_starts = torch.tensor([pad[0], pad[2], pad[4]], device=locations.device)
    return (locations + pad_starts).long()

def location_to_index_3d(locations: Tensor, ny: int, nx: int) -> Tensor:
    """Convert 3D (z, y, x) locations to flat indices for (nz, ny, nx) array."""
    return locations[..., 0] * (ny * nx) + locations[..., 1] * nx + locations[..., 2]



def setup_propagator(
    models: List[Tensor],
    prop_type: str,
    grid_spacing: Union[int, float, List[float], Tensor],
    dt: float,
    wavefields: List[Optional[Tensor]],
    source_amplitudes: List[Optional[Tensor]],
    source_locations: List[Optional[Tensor]],
    receiver_locations: List[Optional[Tensor]],
    accuracy: int = 4,
    pml_width: Union[int, List[int]] = 20,
    pml_freq: Optional[float] = None,
    max_vel: Optional[float] = None,
    survey_pad: Optional[Union[int, List[Optional[int]]]] = None,
    origin: Optional[List[int]] = None,
    nt: Optional[int] = None,
    model_gradient_sampling_interval: int = 1,
    freq_taper_frac: float = 0.0,
    time_pad_frac: float = 0.0,
    time_taper: bool = False
) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor],
           List[Tensor], List[Tensor], float, float, float, int, int, int, int,
           int, List[int]]:
    # scalar, models = [v]
    # scalar_born, models = [v, scatter]
    # elastic, models = [lamb, mu, buoyancy]
    if prop_type == 'scalar':
        max_model_vel = models[0].abs().max().item()
        min_model_vel = models[0].abs().min().item()
        pad_modes = ['replicate']
        fd_pad = accuracy // 2
    elif prop_type == 'scalar_born':
        max_model_vel = models[0].abs().max().item()
        min_model_vel = models[0].abs().min().item()
        pad_modes = ['replicate', 'constant']
        fd_pad = accuracy // 2
    elif prop_type == 'elastic':
        vp, vs, _ = lambmubuoyancy_to_vpvsrho(models[0].abs(), models[1].abs(),
                                              models[2].abs())
        max_model_vel = max(vp.abs().max().item(), vs.abs().max().item())
        min_model_vel = min(vp.abs().min().item(), vs.abs().min().item())
        pad_modes = ['replicate', 'replicate', 'replicate']
        fd_pad = 0
    else:
        raise RuntimeError("Unknown prop_type.")

    # 3D: pml_width, pad, etc. should be 6 elements (z, y, x)
    if isinstance(pml_width, int):
        pml_width = [pml_width for _ in range(6)]

    # Check inputs
    check_inputs(source_amplitudes, source_locations, receiver_locations,
                 wavefields, accuracy, nt, models, pml_width, fd_pad, ndim=3)

    if nt is None:
        nt = 0
        for source_amplitude in source_amplitudes:
            if source_amplitude is not None:
                nt = source_amplitude.shape[-1]
                break
    device = models[0].device
    dtype = models[0].dtype
    dz, dy, dx = set_dx(grid_spacing, ndim=3)
    pad = [fd_pad + width for width in pml_width]
    # 只支持 3D: extract_survey、pad、locations、ndim=3
    models, locations = extract_survey(models,
                                       source_locations + receiver_locations,
                                       survey_pad, wavefields, origin,
                                       pml_width, ndim=3)
    source_locations = locations[:len(source_locations)]
    receiver_locations = locations[len(source_locations):]
    if max_vel is None:
        max_vel = max_model_vel
    else:
        max_vel = abs(max_vel)
        if max_vel < max_model_vel:
            warnings.warn("max_vel is less than the actual maximum velocity.")
    check_points_per_wavelength(min_model_vel, pml_freq, dz, dy, dx, dt = dt)
    for i in range(len(models)):
        models[i] = pad_model_3d(models[i], pad, mode=pad_modes[i])
    nz, ny, nx = models[0].shape
    dt, step_ratio = cfl_condition(dz, dy, dx, dt, max_vel)
    sources_i: List[Optional[Tensor]] = []
    for i in range(len(source_amplitudes)):
        sa = source_amplitudes[i]
        sl = source_locations[i]
        if sa is not None and sl is not None:
            sl = pad_locations_3d(sl, pad)
            sources_i.append(location_to_index_3d(sl, ny, nx))
            if prop_type == 'scalar':
                sa = (-sa * models[0][sl[..., 0], sl[..., 1], sl[..., 2], None]**2 * dt**2)
            elif prop_type == 'scalar_born':
                if i == 0:  # background
                    sa = (-sa * models[0][sl[..., 0], sl[..., 1], sl[..., 2], None]**2 * dt**2)
                else:  # scattered
                    sa = (-2 * sa * models[0][sl[..., 0], sl[..., 1], sl[..., 2], None] *
                          models[1][sl[..., 0], sl[..., 1], sl[..., 2], None] * dt**2)
            elif prop_type == 'elastic':
                # 3D elastic not implemented here
                raise NotImplementedError("3D elastic source injection not implemented.")
            sa = upsample(sa,
                          step_ratio,
                          freq_taper_frac=freq_taper_frac,
                          time_pad_frac=time_pad_frac,
                          time_taper=time_taper)
            if sa.device == torch.device('cpu'):
                sa = torch.movedim(sa, -1, 1)
            else:
                sa = torch.movedim(sa, -1, 0)
            source_amplitudes[i] = sa
        else:
            sources_i.append(None)
    receivers_i: List[Optional[Tensor]] = []
    for i, rl in enumerate(receiver_locations):
        if rl is not None:
            rl = pad_locations_3d(rl, pad)
            receivers_i.append(location_to_index_3d(rl, ny, nx))
        else:
            receivers_i.append(None)
    n_batch = get_n_batch(source_locations + receiver_locations +
                          list(wavefields))
    # 3D PML profiles (z, y, x)
    az, bz = setup_pml(pml_width[:2], [fd_pad+1, fd_pad+1], dt, dz, nz, max_vel, dtype, device, pml_freq)
    ay, by = setup_pml(pml_width[2:4], [fd_pad+1, fd_pad+1], dt, dy, ny, max_vel, dtype, device, pml_freq)
    ax, bx = setup_pml(pml_width[4:], [fd_pad+1, fd_pad+1], dt, dx, nx, max_vel, dtype, device, pml_freq)
    # Zero out PML at interface
    az[fd_pad + pml_width[0]] = 0
    az[-(fd_pad + pml_width[1] + 1)] = 0
    ay[fd_pad + pml_width[2]] = 0
    ay[-(fd_pad + pml_width[3] + 1)] = 0
    ax[fd_pad + pml_width[4]] = 0
    ax[-(fd_pad + pml_width[5] + 1)] = 0
    pml_profiles = [az, ay, ax, bz, by, bx]
    nt *= step_ratio

    for i in range(len(models)):
        models[i] = convert_to_contiguous(models[i])
    contiguous_source_amplitudes: List[Tensor] = []
    for i in range(len(source_amplitudes)):
        contiguous_source_amplitudes.append(
            convert_to_contiguous(source_amplitudes[i]).to(dtype).to(device))
    contiguous_wavefields = []
    for wavefield in wavefields:
        contiguous_wavefields.append(
            convert_to_contiguous(wavefield).to(dtype).to(device))
    contiguous_sources_i: List[Tensor] = []
    for i in range(len(sources_i)):
        contiguous_sources_i.append(convert_to_contiguous(sources_i[i]))
    contiguous_receivers_i: List[Tensor] = []
    for i in range(len(receivers_i)):
        contiguous_receivers_i.append(convert_to_contiguous(receivers_i[i]))
    for i in range(len(pml_profiles)):
        pml_profiles[i] = convert_to_contiguous(pml_profiles[i])
    return (models, contiguous_source_amplitudes, contiguous_wavefields,
            pml_profiles, contiguous_sources_i, contiguous_receivers_i, dz, dy, dx,
            dt, nt, n_batch, step_ratio, model_gradient_sampling_interval,
            accuracy, pml_width)


def downsample_and_movedim(receiver_amplitudes: Tensor,
                           step_ratio: int,
                           freq_taper_frac: float = 0.0,
                           time_pad_frac: float = 0.0,
                           time_taper: bool = False,
                           shift: float = 0.0) -> Tensor:
    if receiver_amplitudes.numel() > 0:
        if receiver_amplitudes.device == torch.device('cpu'):
            receiver_amplitudes = torch.movedim(receiver_amplitudes, 1, -1)
        else:
            receiver_amplitudes = torch.movedim(receiver_amplitudes, 0, -1)
        receiver_amplitudes = downsample(receiver_amplitudes,
                                         step_ratio,
                                         freq_taper_frac=freq_taper_frac,
                                         time_pad_frac=time_pad_frac,
                                         time_taper=time_taper,
                                         shift=shift)
    return receiver_amplitudes


def set_dx(dx: Union[int, float, List[float], Tensor], ndim: int = 2) -> Tuple:
    """Return grid spacing for each dimension (supports 2D/3D)."""
    if isinstance(dx, (int, float)):
        return tuple([float(dx)] * ndim)
    if isinstance(dx, list):
        if len(dx) == ndim:
            return tuple(float(d) for d in dx)
    if isinstance(dx, torch.Tensor) and dx.numel() == ndim:
        return tuple(float(d.item()) for d in dx)
    raise RuntimeError(f"Expected dx to be a real number or a list/tensor of length {ndim}.")


def check_inputs(source_amplitudes: List[Optional[Tensor]],
                 source_locations: List[Optional[Tensor]],
                 receiver_locations: List[Optional[Tensor]],
                 wavefields: List[Optional[Tensor]], accuracy: int,
                 nt: Optional[int], models: List[Tensor], pml_width: List[int],
                 fd_pad: int, ndim: int = 2) -> None:
    device = models[0].device
    dtype = models[0].dtype
    if len(source_amplitudes) != len(source_locations):
        raise RuntimeError("The same number of source_amplitudes and "
                           "source_locations must be provided.")
    for i in range(len(source_amplitudes)):
        if source_amplitudes[i] is not None or source_locations[i] is not None:
            if source_amplitudes[i] is None or source_locations[i] is None:
                raise RuntimeError("source_locations and source_amplitudes must both be None or non-None.")
        source_amplitudesi = source_amplitudes[i]
        source_locationsi = source_locations[i]
        if (source_amplitudesi is not None and source_locationsi is not None):
            if source_amplitudesi.ndim != 3:
                raise RuntimeError("source_amplitudes should have dimensions [shot, source, time].")
            if source_locationsi.ndim != 3:
                raise RuntimeError(f"source_locations should have dimensions [shot, source, dimension={ndim}].")
            if source_amplitudesi.shape[0] != source_locationsi.shape[0]:
                raise RuntimeError("Expected source_amplitudes.shape[0] == source_locations.shape[0].")
            if source_amplitudesi.shape[1] != source_locationsi.shape[1]:
                raise RuntimeError("Expected source_amplitudes.shape[1] == source_locations.shape[1].")
            if source_locationsi.shape[2] != ndim:
                raise RuntimeError(f"Expected source_locations.shape[2] == {ndim}.")
            if source_amplitudesi.device != device:
                raise RuntimeError("Expected source_amplitudes and the model to be on the same device.")
            if source_locationsi.device != device:
                raise RuntimeError("Expected source_locations and the model to be on the same device.")
            if source_amplitudesi.dtype != dtype:
                raise RuntimeError("Expected source_amplitudes and the model to have the same dtype.")
            if source_amplitudesi.numel() == 0:
                raise RuntimeError("source_amplitudes has zero elements. Please instead set it to None if you do not wish to use a source.")
    sa_nt = -1
    for sa in source_amplitudes:
        if sa is not None:
            if sa_nt < 0:
                sa_nt = sa.shape[-1]
            else:
                if sa.shape[-1] != sa_nt:
                    raise RuntimeError("All source_amplitudes must have "
                                       "the same length in the time "
                                       "dimension.")
            if nt is not None:
                raise RuntimeError("Specify source_amplitudes or nt, "
                                   "not both.")
    if sa_nt < 0 and nt is None:
        raise RuntimeError("Specify source_amplitudes or nt.")

    for receiver_location in receiver_locations:
        if receiver_location is not None:
            if receiver_location.ndim != 3:
                raise RuntimeError("receiver_locations should have dimensions [shot, receiver, dimension=3].")
            if receiver_location.shape[2] != 3:
                raise RuntimeError("Expected receiver_locations.shape[2] == 3 for 3D.")
            if receiver_location.device != device:
                raise RuntimeError("Expected receiver_locations and the model to be on the same device.")
            if receiver_location.numel() == 0:
                raise RuntimeError("receiver_locations has zero elements. Please instead set it to None if you do not wish to have receivers.")

    n_shots = -1
    for source_location in source_locations:
        if source_location is not None:
            if n_shots < 0:
                n_shots = source_location.shape[0]
            else:
                if source_location.shape[0] != n_shots:
                    raise RuntimeError("Expected source_locations and "
                                       "receiver_locations to all have the "
                                       "same number of shots.")
    for receiver_location in receiver_locations:
        if receiver_location is not None:
            if n_shots < 0:
                n_shots = receiver_location.shape[0]
            else:
                if receiver_location.shape[0] != n_shots:
                    raise RuntimeError("Expected source_locations and "
                                       "receiver_locations to all have the "
                                       "same number of shots.")


    wavefield_shape = [0] * (ndim + 1)
    for wavefield in wavefields:
        if wavefield is not None:
            if wavefield.ndim != ndim + 1:
                raise RuntimeError(f"All wavefields must have {ndim+1} dimensions.")
            if wavefield_shape == [0] * (ndim + 1):
                wavefield_shape = [wavefield.shape[i] for i in range(ndim + 1)]
            else:
                if any(wavefield_shape[i] != wavefield.shape[i] for i in range(ndim + 1)):
                    raise RuntimeError("All wavefields must have the same shape.")
            if wavefield.device != device:
                raise RuntimeError("All wavefields must be on the same device as the model.")
            if wavefield.dtype != dtype:
                raise RuntimeError("All wavefields must have the same dtype as the model.")
    if (n_shots > 0 and wavefield_shape[0] > 0 and n_shots != wavefield_shape[0]):
        raise RuntimeError("The wavefield batch dimension does not match the source/receiver batch dimension.")

    if accuracy not in [2, 4, 6, 8]:
        raise RuntimeError("Only accuracy values of 2, 4, 6, and 8 are implemented.")

    if models[0].ndim != ndim:
        raise RuntimeError(f"Models must have {ndim} dimensions.")
    model_shape = models[0].shape
    for model in models[1:]:
        if model.device != device:
            raise RuntimeError("All models must be on the same device.")
        if model.dtype != dtype:
            raise RuntimeError("All models must have the same dtype.")
        if model.ndim != ndim:
            raise RuntimeError(f"Models must have {ndim} dimensions.")
        if any(model.shape[i] != model_shape[i] for i in range(ndim)):
            raise RuntimeError("All models must have the same shape.")

    if not isinstance(pml_width, list):
        raise RuntimeError("pml_width must be a list of ints.")
    if any([not isinstance(dim_pad, int) for dim_pad in pml_width]):
        raise RuntimeError("pml_width must be a list of ints.")
    if any([dim_pad < 0 for dim_pad in pml_width]):
        raise RuntimeError("pml_width must be positive.")
    if len(pml_width) != 2 * ndim:
        raise RuntimeError(f"pml_width must be a list of length {2*ndim}.")

    for d in range(ndim):
        if wavefield_shape[d+1] > model_shape[d] + pml_width[2*d] + pml_width[2*d+1]:
            raise RuntimeError(f"Provided wavefields must be of size at most {model_shape[d] + pml_width[2*d] + pml_width[2*d+1]} in spatial dimension {d}, but is {wavefield_shape[d+1]}")


def check_points_per_wavelength(min_vel: float, pml_freq: Optional[float],
                                *spacings: float, dt: float) -> None:
    """Check grid points per wavelength for 2D/3D, warn if <6."""
    if pml_freq is not None:
        min_wavelength = min_vel / pml_freq
        max_spacing = max(spacings)
        if min_wavelength / max_spacing < 6:
            warnings.warn(
                f"At least six grid cells per wavelength is recommended, but at a frequency of {pml_freq}, a minimum velocity of {min_vel}, and a grid cell spacing of {max_spacing}, there are only {min_wavelength / max_spacing}."
            )
        if 1 / 2 / dt < pml_freq:
            warnings.warn("The provided pml_freq is greater than the Nyquist frequency of the data.")


def get_n_batch(args: List[Optional[Tensor]]) -> int:
    "Get the size of the batch dimension."
    for arg in args:
        if arg is not None:
            return arg.shape[0]
    raise RuntimeError("One of the inputs to get_n_batch must be non-None.")


def pad_model(model: Tensor,
              pad: List[int],
              mode: str = "replicate") -> Tensor:
    # 只支持 3D: model (nz, ny, nx), pad [z0, z1, y0, y1, x0, x1]
    if model.ndim == 3 and len(pad) == 6:
        return torch.nn.functional.pad(model.unsqueeze(0).unsqueeze(0),
                                       (pad[4], pad[5], pad[2], pad[3], pad[0], pad[1]),
                                       mode=mode)[0, 0]
    else:
        raise RuntimeError("pad_model: only supports 3D (ndim=3, pad=6).")


def pad_locations(locations: Tensor, pad: List[int]) -> Tensor:
    # 只支持 3D: locations[...,3], pad[6]
    if locations.shape[-1] == 3 and len(pad) == 6:
        pad_starts = [pad[0], pad[2], pad[4]]
    else:
        raise RuntimeError("pad_locations: only supports 3D (locations[...,3], pad=6).")
    pad_starts_tensor = torch.tensor(pad_starts).to(locations.device)
    return (locations + pad_starts_tensor).long()


def location_to_index(locations: Tensor, *shape) -> Tensor:
    # 只支持 3D: (nz, ny, nx)
    if locations.shape[-1] == 3 and len(shape) == 2:
        ny, nx = shape
        return locations[..., 0] * (ny * nx) + locations[..., 1] * nx + locations[..., 2]
    else:
        raise RuntimeError("location_to_index: only supports 3D (locations[...,3], shape=(ny, nx)).")


def cosine_taper_end(signal: Tensor, n_taper: int) -> Tensor:
    """Tapers the end of the final dimension of a Tensor using a cosine.

    A half period, shifted and scaled to taper from 1 to 0, is used.

    Args:
        signal:
            The Tensor that will have its final dimension tapered.
        n_taper:
            The length of the cosine taper, in number of samples.

    Returns:
        The tapered signal.
    """
    taper = torch.ones(signal.shape[-1],
                       dtype=signal.dtype,
                       device=signal.device)
    taper[len(taper) - n_taper:] = (torch.cos(
        torch.arange(1, n_taper + 1, device=signal.device) /
        n_taper * math.pi) + 1).to(signal.dtype) / 2
    return signal * taper


def zero_last_element_of_final_dimension(signal: Tensor) -> Tensor:
    zeroer = torch.ones(signal.shape[-1],
                        dtype=signal.dtype,
                        device=signal.device)
    zeroer[-1] = 0
    return signal * zeroer


def upsample(signal: Tensor,
             step_ratio: int,
             freq_taper_frac: float = 0.0,
             time_pad_frac: float = 0.0,
             time_taper: bool = False) -> Tensor:
    """Upsamples the final dimension of a Tensor by a factor.

    Low-pass upsampling is used to produce an upsampled signal without
    introducing higher frequencies than were present in the input. The
    Nyquist frequency of the input will be zeroed.

    Args:
        signal:
            The Tensor that will have its final dimension upsampled.
        step_ratio:
            The integer factor by which the signal will be upsampled.
            The input signal is returned if this is 1 (freq_taper_frac,
            time_pad_frac, and time_taper will be ignored).
        freq_taper_frac:
            A float specifying the fraction of the end of the signal
            amplitude in the frequency domain to cosine taper. This
            might be useful to reduce ringing. A value of 0.1 means
            that the top 10% of frequencies will be tapered before
            upsampling. Default 0.0 (no tapering).
        time_pad_frac:
            A float specifying the amount of zero padding that will
            be added to the signal before upsampling and removed
            afterwards, as a fraction of the length of the final
            dimension of the input signal. This might be useful to reduce
            wraparound artifacts. A value of 0.1 means that zero padding
            of 10% of the length of the signal will be used. Default 0.0.
        time_taper:
            A bool specifying whether to apply a Hann window in time.
            This is useful during correctness tests of the propagators
            as it ensures that signals taper to zero at their edges in
            time, avoiding the possibility of high frequencies being
            introduced.

    Returns:
        The signal after upsampling.
    """
    if step_ratio == 1:
        return signal
    if time_pad_frac > 0.0:
        n_time_pad = int(time_pad_frac * signal.shape[-1])
        signal = torch.nn.functional.pad(signal, (0, n_time_pad))
    else:
        n_time_pad = 0
    nt = signal.shape[-1]
    up_nt = nt * step_ratio
    signal_f = torch.fft.rfft(signal, norm='ortho') * math.sqrt(step_ratio)
    if freq_taper_frac > 0.0:
        freq_taper_len = int(freq_taper_frac * signal_f.shape[-1])
        signal_f = cosine_taper_end(signal_f, freq_taper_len)
    elif signal_f.shape[-1] > 1:
        # Set Nyquist frequency to zero
        signal_f = zero_last_element_of_final_dimension(signal_f)
    signal_f = torch.nn.functional.pad(signal_f,
                                       (0, up_nt // 2 + 1 - nt // 2 + 1))
    signal = torch.fft.irfft(signal_f, n=up_nt, norm='ortho')
    if time_pad_frac > 0.0:
        signal = signal[..., :signal.shape[-1] - n_time_pad * step_ratio]
    if time_taper:
        signal = signal * torch.hann_window(signal.shape[-1], periodic=False,
                                            device=signal.device)
    return signal


def downsample(signal: Tensor,
               step_ratio: int,
               freq_taper_frac: float = 0.0,
               time_pad_frac: float = 0.0,
               time_taper: bool = False,
               shift: float = 0.0) -> Tensor:
    """Downsamples the final dimension of a Tensor by a factor.

    Frequencies higher than or equal to the Nyquist frequency of the
    downsampled signal will be zeroed before downsampling.

    Args:
        signal:
            The Tensor that will have its final dimension downsampled.
        step_ratio:
            The integer factor by which the signal will be downsampled.
            The input signal is returned if this is 1 and shift is 0
            (freq_taper_frac, time_pad_frac, and time_taper will be
            ignored).
        freq_taper_frac:
            A float specifying the fraction of the end of the signal
            amplitude in the frequency domain to cosine taper. This
            might be useful to reduce ringing. A value of 0.1 means
            that the top 10% of frequencies will be tapered after
            downsampling. Default 0.0 (no tapering).
        time_pad_frac:
            A float specifying the amount of zero padding that will
            be added to the signal before downsampling and removed
            afterwards, as a fraction of the length of the final
            dimension of the output signal. This might be useful to reduce
            wraparound artifacts. A value of 0.1 means that zero padding
            of 10% of the length of the output signal will be used.
            Default 0.0.
        time_taper:
            A bool specifying whether to apply a Hann window in time.
            This is useful during correctness tests of the propagators
            as it ensures that signals taper to zero at their edges in
            time, avoiding the possibility of high frequencies being
            introduced.
        shift:
            Amount (in units of time samples) to shift the data in time
            before downsampling.
            Default 0.0.

    Returns:
        The signal after downsampling.
    """
    if step_ratio == 1 and shift == 0.0:
        return signal
    if time_taper:
        signal = signal * torch.hann_window(signal.shape[-1], periodic=False,
                                            device=signal.device)
    if time_pad_frac > 0.0:
        n_time_pad = int(time_pad_frac * (signal.shape[-1] // step_ratio))
        signal = torch.nn.functional.pad(signal, (0, n_time_pad * step_ratio))
    else:
        n_time_pad = 0
    nt = signal.shape[-1]
    down_nt = nt // step_ratio
    signal_f = torch.fft.rfft(signal, norm='ortho')[..., :down_nt // 2 + 1]
    if freq_taper_frac > 0.0:
        freq_taper_len = int(freq_taper_frac * signal_f.shape[-1])
        signal_f = cosine_taper_end(signal_f, freq_taper_len)
    elif signal_f.shape[-1] > 1:
        # Set Nyquist frequency to zero
        signal_f = zero_last_element_of_final_dimension(signal_f)
    if shift != 0.0:
        signal_f *= torch.exp(-1j * 2 * math.pi * torch.fft.rfftfreq(
            signal.shape[-1], device=signal.device)[:down_nt // 2 + 1] * shift)
    signal = torch.fft.irfft(signal_f, n=down_nt,
                             norm='ortho') / math.sqrt(step_ratio)
    if time_pad_frac > 0.0:
        signal = signal[..., :signal.shape[-1] - n_time_pad]
    return signal


def convert_to_contiguous(tensor: Optional[Tensor]) -> Tensor:
    if tensor is None:
        return torch.empty(0)
    return tensor.contiguous()


def pad_wavefield(tensor: Optional[Tensor], shape: List[int], fd_pad: int,
                  dtype: torch.dtype, device: torch.device) -> Tensor:
    """Pad wavefield to shape with fd_pad for 2D/3D."""
    if tensor is None:
        return torch.zeros(*shape, dtype=dtype, device=device)
    if tensor.ndim == 4 and len(shape) == 4:
        # 3D: (batch, nz, ny, nx)
        return torch.nn.functional.pad(tensor, (fd_pad, fd_pad, fd_pad, fd_pad, fd_pad, fd_pad)).contiguous()
    else:
        raise RuntimeError("pad_wavefield: only supports 3D (ndim=4, len(shape)=4).")


def extract_survey(
        models: List[Tensor], locations: List[Optional[Tensor]],
        survey_pad: Optional[Union[int, List[Optional[int]]]],
        wavefields: List[Optional[Tensor]], origin: Optional[List[int]],
        pad: List[int], ndim: int = 2) -> Tuple[List[Tensor], List[Optional[Tensor]]]:
    """Extract survey region for 3D."""
    if survey_pad is not None and origin is not None:
        raise RuntimeError("survey_pad and origin cannot both be specified.")
    check_locations_are_within_model(models[0].shape, locations)
    if (survey_pad is None and any([wavefield is not None for wavefield in wavefields])):
        survey_extents = get_survey_extents_from_wavefields(wavefields, origin, pad)
        check_extents_within_model(survey_extents, models[0].shape)
        check_locations_within_extents(survey_extents, locations)
    else:
        survey_extents = get_survey_extents_from_locations(models[0].shape, locations, survey_pad, ndim=3)
        check_extents_match_wavefields_shape(survey_extents, wavefields, pad)
    return extract_models(models, survey_extents, ndim=3), extract_locations(locations, survey_extents)


def check_locations_are_within_model(
        model_shape: List[int], locations: List[Optional[Tensor]]) -> None:
    # 只支持 3D: model_shape=[nz,ny,nx], location[...,3]
    for location in locations:
        if location is not None:
            if location.shape[-1] != 3:
                raise RuntimeError("Location tensor must have last dim=3 for 3D.")
            if location.min() < 0:
                raise RuntimeError("Locations must be >= 0.")
            if location[..., 0].max() >= model_shape[0]:
                raise RuntimeError("z index out of model bounds.")
            if location[..., 1].max() >= model_shape[1]:
                raise RuntimeError("y index out of model bounds.")
            if location[..., 2].max() >= model_shape[2]:
                raise RuntimeError("x index out of model bounds.")


def set_survey_pad(survey_pad: Optional[Union[int, List[Optional[int]]]],
                   ndim: int) -> List[int]:
    """Check survey_pad, and convert to a list if it is a scalar."""
    # 只支持 3D: ndim=3
    if survey_pad is None:
        return [-1] * 6
    elif isinstance(survey_pad, int):
        if survey_pad < 0:
            raise RuntimeError("survey_pad must be non-negative.")
        return [survey_pad] * 6
    elif isinstance(survey_pad, list):
        survey_pad_list = []
        for pad in survey_pad:
            if pad is None:
                survey_pad_list.append(-1)
            elif pad >= 0:
                survey_pad_list.append(pad)
            else:
                raise RuntimeError("survey_pad must be non-negative.")
        if len(survey_pad_list) != 6:
            raise RuntimeError("survey_pad must have length 6 for 3D.")
        return survey_pad_list
    else:
        raise RuntimeError("survey_pad must be None, an int, or a list.")


def get_survey_extents_from_locations(
        model_shape: List[int], locations: List[Optional[Tensor]],
        survey_pad: Optional[Union[int, List[Optional[int]]]], ndim: int = 2) -> List[int]:
    """Calculate the extents of the model to use for the survey (2D/3D)."""
    # 只支持 3D: model_shape=[nz,ny,nx], survey_pad_list=[z0,z1,y0,y1,x0,x1]
    survey_pad_list = set_survey_pad(survey_pad, 3)
    extents: List[int] = []
    # z
    left_z = get_survey_extents_one_side(survey_pad_list[0], "left", 0, locations, model_shape[0])
    right_z = get_survey_extents_one_side(survey_pad_list[1], "right", 0, locations, model_shape[0])
    extents.extend([left_z, right_z])
    # y
    left_y = get_survey_extents_one_side(survey_pad_list[2], "left", 1, locations, model_shape[1])
    right_y = get_survey_extents_one_side(survey_pad_list[3], "right", 1, locations, model_shape[1])
    extents.extend([left_y, right_y])
    # x
    left_x = get_survey_extents_one_side(survey_pad_list[4], "left", 2, locations, model_shape[2])
    right_x = get_survey_extents_one_side(survey_pad_list[5], "right", 2, locations, model_shape[2])
    extents.extend([left_x, right_x])
    return extents


def get_survey_extents_one_side(pad: int, side: str, dim: int,
                                locations: List[Optional[Tensor]],
                                shape: int) -> int:
    """Get the survey extent for the left or right side of one dimension.

    Args:
        pad:
            Positive int specifying padding for the side, or negative if
            the extracted model should continue to the edge of the model.
        side:
            'left' or 'right'.
        dim:
            Integer specifying the dimension index.
        locations:
            List of Tensors with coordinates for the current dimension.
        shape:
            Int specifying length of full model in current dimension.

    Returns:
        Min/max index as int.
    """
    # 只支持 3D
    if side == "left":
        if pad < 0 or not any([location is not None for location in locations]):
            return 0
        extreme_location = shape
        for location in locations:
            if location is not None:
                extreme_location = min(extreme_location, int(location[..., dim].min().item()) - pad)
    else:
        if pad < 0 or not any([location is not None for location in locations]):
            return shape
        extreme_location = 0
        for location in locations:
            if location is not None:
                extreme_location = max(extreme_location, int(location[..., dim].max().item()) + pad)
        extreme_location += 1
    extreme_location = max(extreme_location, 0)
    extreme_location = min(extreme_location, shape)
    return extreme_location


def get_survey_extents_from_wavefields(wavefields: List[Optional[Tensor]],
                                       origin: Optional[List[int]],
                                       pad: List[int]) -> List[int]:
    """Determine the extent of the model to extract from the wavefields.

    Args:
        wavefields:
            A list of initial wavefields (potentially None).
        origin:
            A list containing the coordinates of the origin of the wavefields.
            Optional, default [0, 0].
        pad:
            A list containing the PML width of each side.

    Returns:
        A list of length twice the number of dimensions in the model,
        specifying the extents of the model that will be
        used for wave propagation.
    """

    if origin is not None:
        if any([not isinstance(dim_origin, int) for dim_origin in origin]):
            raise RuntimeError("origin must be list of ints.")
        if any([dim_origin < 0 for dim_origin in origin]):
            raise RuntimeError("origin coordinates must be non-negative.")
    if not isinstance(pad, list):
        raise RuntimeError("pad must be a list of ints.")
    if any([not isinstance(dim_pad, int) for dim_pad in pad]):
        raise RuntimeError("pad must be a list of ints.")
    if any([dim_pad < 0 for dim_pad in pad]):
        raise RuntimeError("pad must be positive.")
    # 只支持 3D: pad=[z0,z1,y0,y1,x0,x1], origin=[z,y,x]
    ndims = 3
    extents: List[int] = []
    for wavefield in wavefields:
        if wavefield is not None:
            if wavefield.ndim != 4:
                raise RuntimeError("wavefields must have 4 dims (batch, z, y, x) for 3D.")
            if origin is not None:
                if len(origin) != 3:
                    raise RuntimeError("origin must have 3 elements for 3D.")
                extents.append(origin[0])
                extents.append(origin[0] + wavefield.shape[1] - (pad[0] + pad[1]))
                extents.append(origin[1])
                extents.append(origin[1] + wavefield.shape[2] - (pad[2] + pad[3]))
                extents.append(origin[2])
                extents.append(origin[2] + wavefield.shape[3] - (pad[4] + pad[5]))
            else:
                extents.append(0)
                extents.append(wavefield.shape[1] - (pad[0] + pad[1]))
                extents.append(0)
                extents.append(wavefield.shape[2] - (pad[2] + pad[3]))
                extents.append(0)
                extents.append(wavefield.shape[3] - (pad[4] + pad[5]))
            return extents
    raise RuntimeError("At least one wavefield must be non-None.")


def check_extents_within_model(extents: List[int],
                               model_shape: List[int]) -> None:
    for (extent, model_dim_shape) in zip(extents[::2], model_shape):
        if extent > model_dim_shape:
            raise RuntimeError("Wavefields must be within the model.")


def check_locations_within_extents(extents: List[int],
                                   locations: List[Optional[Tensor]]) -> None:
    for location in locations:
        if location is not None:
            for dim in range(location.shape[-1]):
                if (location is not None and
                    (location[..., dim].min() < extents[dim * 2]
                     or location[..., dim].max() >= extents[dim * 2 + 1])):
                    raise RuntimeError("Locations must be within "
                                       "wavefields.")


def check_extents_match_wavefields_shape(extents: List[int],
                                         wavefields: List[Optional[Tensor]],
                                         pad: List[int]):
    # 只支持 3D: 检查 (z, y, x) 维度
    for wavefield in wavefields:
        if wavefield is not None:
            sz = extents[1] - extents[0] + pad[0] + pad[1]
            sy = extents[3] - extents[2] + pad[2] + pad[3]
            sx = extents[5] - extents[4] + pad[4] + pad[5]
            if (wavefield.shape[1] != sz or wavefield.shape[2] != sy or wavefield.shape[3] != sx):
                raise RuntimeError(
                    f"The shape of the provided wavefield {wavefield.shape[1:]} does not match the extent determined by the specified survey_pad, expected (z,y,x)=({sz},{sy},{sx})")


def extract_models(models: List[Tensor], extents: List[int], ndim: int = 2) -> List[Tensor]:
    """Extract the specified portion of the model (3D only)."""
    # 只支持 3D，extents: [z0, z1, y0, y1, x0, x1]
    return [model[extents[0]:extents[1], extents[2]:extents[3], extents[4]:extents[5]] for model in models]


def extract_locations(locations: List[Optional[Tensor]],
                      extents: List[int]) -> List[Optional[Tensor]]:
    """Set locations relative to extracted 3D model."""
    # extents: [z0, z1, y0, y1, x0, x1]
    origin = [extents[0], extents[2], extents[4]]
    extracted_locations: List[Optional[Tensor]] = []
    for location in locations:
        if location is not None:
            extracted_locations.append(
                location - torch.tensor(origin, device=location.device))
        else:
            extracted_locations.append(location)
    return extracted_locations


def cfl_condition(dz: float,
                  dy: float,
                  dx: float,
                  dt: float,
                  max_vel: float,
                  eps: float = 1e-15) -> Tuple[float, int]:
    """Calculates the time step interval to obey the CFL condition.

    The output time step will be a factor of the input time step.

    Args:
        dy:
            The grid spacing in the first dimension.
        dx:
            The grid spacing in the second dimension.
        dt:
            The time step interval.
        max_vel:
            The maximum wavespeed in the model.
        eps:
            A small quantity to prevent division by zero. Default 1e-15.

    Returns:
        Tuple[float, int]:

            inner_dt:
                A time step interval that obeys the CFL condition.
            step_ratio:
                The integer dt / inner_dt.
    """
    # 仅保留 2D 兼容性，3D 推荐用 cfl_condition_3d
    max_dt = (0.6 / math.sqrt(1 / dz**2 + 1 / dy**2 + 1 / dx**2) /
              (max_vel**2 + eps)) * max_vel
    step_ratio = int(math.ceil(abs(dt) / max_dt))
    inner_dt = dt / step_ratio
    return inner_dt, step_ratio


def vpvsrho_to_lambmubuoyancy(
        vp: Tensor,
        vs: Tensor,
        rho: Tensor,
        eps: float = 1e-15) -> Tuple[Tensor, Tensor, Tensor]:
    """Converts vp, vs, rho to lambda, mu, buoyancy.

    All input Tensors must have the same shape.

    Args:
        vp:
            A Tensor containing the p wavespeed.
        vs:
            A Tensor containing the s wavespeed.
        rho:
            A Tensor containing the density.
        eps:
            An optional float to avoid division by zero. Default 1e-15.

    Returns:
        Tuple[Tensor, Tensor, Tensor]:

            lambda:
                A Tensor containing the first Lamé parameter.
            mu:
                A Tensor containing the second Lamé parameter.
            buoyancy:
                A Tensor containing the reciprocal of density.
    """

    # 3D: 支持任意 shape，直接广播
    lamb = (vp**2 - 2 * vs**2) * rho
    mu = vs**2 * rho
    buoyancy = 1 / (rho**2 + eps) * rho
    return lamb, mu, buoyancy


def lambmubuoyancy_to_vpvsrho(
        lamb: Tensor,
        mu: Tensor,
        buoyancy: Tensor,
        eps: float = 1e-15) -> Tuple[Tensor, Tensor, Tensor]:
    """Converts lambda, mu, buoyancy to vp, vs, rho.

    All input Tensors must have the same shape.

    Args:
        lambda:
            A Tensor containing the first Lamé parameter.
        mu:
            A Tensor containing the second Lamé parameter.
        buoyancy:
            A Tensor containing the reciprocal of density.
        eps:
            An optional float to avoid division by zero. Default 1e-15.

    Returns:
        Tuple[Tensor, Tensor, Tensor]:

            vp:
                A Tensor containing the p wavespeed.
            vs:
                A Tensor containing the s wavespeed.
            rho:
                A Tensor containing the density.
    """
    # 3D: 支持任意 shape，直接广播
    vs = (mu * buoyancy).sqrt()
    vp = (lamb * buoyancy + 2 * vs**2).sqrt()
    rho = 1 / (buoyancy**2 + eps) * buoyancy
    return vp, vs, rho


def setup_pml(pml_width: List[int],
              fd_pads: List[int],
              dt: float,
              dx: float,
              n: int,
              max_vel: float,
              dtype: torch.dtype,
              device: torch.device,
              pml_freq: Optional[float],
              left_start: float = 0.0,
              right_start: float = 0.0,
              eps: float = 1e-9) -> Tuple[Tensor, Tensor]:
    # 3D: 每个方向分别调用，pml_width/ fd_pads/ dx/ n 均为单方向参数
    R = 0.001
    if pml_freq is None:
        pml_freq = 25.0
    alpha0 = math.pi * pml_freq
    a = torch.zeros(n, device=device, dtype=dtype)
    b = torch.zeros(n, device=device, dtype=dtype)
    for side, (width, fd_pad) in enumerate(zip(pml_width, fd_pads)):
        if width == 0:
            continue
        if side == 0:
            start = left_start
        else:
            start = right_start
        pml_frac = (start + torch.arange(width + fd_pad, device=device, dtype=dtype)) / max(width, 1)
        sigma = -3 * max_vel * math.log(R) / (2 * max(width, 1) * dx) * pml_frac**2
        alpha = alpha0 * (1 - pml_frac)
        sigmaalpha = sigma + alpha
        a_side = torch.exp(-sigmaalpha * abs(dt))
        b_side = sigma / (sigmaalpha**2 + eps) * sigmaalpha * (a_side - 1)
        if side == 0:
            a[:len(pml_frac)] = a_side.flip(0)
            b[:len(pml_frac)] = b_side.flip(0)
        else:
            a[-len(pml_frac):] = a_side
            b[-len(pml_frac):] = b_side
    return a, b


def create_or_pad(tensor: Tensor, fd_pad: int, device: torch.device,
                  dtype: torch.dtype, size: List[int]) -> Tensor:
    if tensor.numel() == 0:
        # size: [nz, ny, nx]
        return torch.zeros(size[0], size[1], size[2], size[3], device=device, dtype=dtype)
    elif fd_pad == 0:
        return tensor.clone()
    else:
        # Pad 3D tensor: (nz, ny, nx) with (fd_pad, fd_pad) for each dim
        return (
            torch.nn.functional.pad(
                tensor,
                (fd_pad, fd_pad, fd_pad, fd_pad, fd_pad, fd_pad)
            )
        ).requires_grad_(tensor.requires_grad)


def zero_edges(tensor: Tensor, fd_pad: int) -> None:
    nz = tensor.shape[0]
    ny = tensor.shape[1]
    nx = tensor.shape[2]
    # Zero z edges
    tensor[:fd_pad, :, :].fill_(0)
    tensor[nz - fd_pad:, :, :].fill_(0)
    # Zero y edges
    tensor[:, :fd_pad, :].fill_(0)
    tensor[:, ny - fd_pad:, :].fill_(0)
    # Zero x edges
    tensor[:, :, :fd_pad].fill_(0)
    tensor[:, :, nx - fd_pad:].fill_(0)


def zero_interior(tensor: Tensor, fd_pad: int, pml_width: List[int],
                  y: bool) -> None:
    nz = tensor.shape[0]
    ny = tensor.shape[1]
    nx = tensor.shape[2]
    # pml_width: [z0, z1, y0, y1, x0, x1]
    # y: if True, zero y interior; else zero x interior; for 3D, also allow z
    if y == 'z':
        # Zero interior along z (keep PML and fd_pad at both ends)
        tensor[:, fd_pad + pml_width[0]:nz - pml_width[1] - fd_pad, :, :].fill_(0)
    elif y == 'y':
        tensor[:, :, fd_pad + pml_width[2]:ny - pml_width[3] - fd_pad, :].fill_(0)
    elif y == 'x':
        tensor[:, :, :, fd_pad + pml_width[4]:nx - pml_width[5] - fd_pad].fill_(0)
    else:
        raise ValueError("y argument must be one of 'z', 'y', or 'x' for 3D zero_interior.")


import torch.nn.functional as F

def diff(a: Tensor, accuracy: int, grid_spacing: float) -> Tensor:
    if accuracy == 2:
        a = torch.nn.functional.pad(a, (1, 1))
        return (1 / 2 * (a[2:] - a[:-2])) / grid_spacing
    if accuracy == 4:
        a = torch.nn.functional.pad(a, (2, 2))
        return (8 / 12 * (a[3:-1] - a[1:-3]) + -1 / 12 *
                (a[4:] - a[:-4])) / grid_spacing
    if accuracy == 6:
        a = torch.nn.functional.pad(a, (3, 3))
        return (3 / 4 * (a[4:-2] - a[2:-4]) + -3 / 20 *
                (a[5:-1] - a[1:-5]) + 1 / 60 * (a[6:] - a[:-6])) / grid_spacing
    a = torch.nn.functional.pad(a, (4, 4))
    return (4 / 5 * (a[5:-3] - a[3:-5]) + -1 / 5 *
            (a[6:-2] - a[2:-6]) + 4 / 105 * (a[7:-1] - a[1:-7]) + -1 / 280 *
            (a[8:] - a[:-8])) / grid_spacing


