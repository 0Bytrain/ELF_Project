
from air_ice_marine import (
    calculate_wave_parameters,
    AIR_PARAMS,
    ICE_PARAMS,
    SEA_PARAMS
)

frequency_hz = 30

#这里是使用类似结构体的指向
air_skin_depth = calculate_wave_parameters(
    frequency_hz,
    AIR_PARAMS["eps_r"],
    AIR_PARAMS["mu_r"],
    AIR_PARAMS["sigma_s_m"]
)

#这里是python特有的解包符号**相当于前面的把结构体里的东西全部解包出来
ice_skin_depth = calculate_wave_parameters(
    frequency_hz,
    ICE_PARAMS["eps_r"],
    ICE_PARAMS["mu_r"],
    ICE_PARAMS["sigma_s_m"]
)

sea_skin_depth = calculate_wave_parameters(
    frequency_hz,
    SEA_PARAMS["eps_r"],
    SEA_PARAMS["mu_r"],
    SEA_PARAMS["sigma_s_m"]
)

print ("空气 skin_depth",air_skin_depth["skin_depth"])
print ("冰 skin_depth",ice_skin_depth["skin_depth"])
print ("海洋 skin_depth",sea_skin_depth["skin_depth"])

