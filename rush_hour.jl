using DataFrames, Dates, Random, Statistics, StatsBase
using CategoricalArrays
using CairoMakie

# ----------------------------
# 1) Synthetic "Citi Bike-ish" trip generator
# ----------------------------
function make_trips(n::Int; seed=7)
    Random.seed!(seed)

    # A few NYC-ish station names (fake but plausible)
    stations = [
        "W 33 St & 7 Ave", "E 17 St & Broadway", "Union Sq E & E 16 St",
        "Broadway & W 60 St", "Bedford Ave & N 7 St", "Fulton St & Broadway",
        "W 4 St & 7 Ave S", "Grand St & Greene St", "E 72 St & Park Ave",
        "Vesey Pl & River Terrace"
    ]

    # Generate timestamps across 7 days
    start_date = Date(2026, 1, 12) # any week
    days = start_date:(start_date + Day(6))

    # Helper to sample "rush-hour-biased" hours
    # Morning peak around 8–9, evening peak around 17–18, plus some baseline
    rush_hours = vcat(fill(8, 4), fill(9, 3), fill(17, 4), fill(18, 3), 12, 13, 14, 20, 21)
    hour_weights = countmap(rush_hours)
    hours = collect(keys(hour_weights))
    weights = [hour_weights[h] for h in hours]
    hour_sampler = Weights(weights)

    trip_id         = collect(1:n)
    start_station   = Vector{String}(undef, n)
    end_station     = Vector{String}(undef, n)
    started_at      = Vector{DateTime}(undef, n)
    duration_min    = Vector{Float64}(undef, n)

    for i in 1:n
        day = rand(days)

        # pick hour with rush-hour bias
        h = sample(hours, hour_sampler)
        m = rand(0:59)
        s = rand(0:59)

        started = DateTime(day) + Hour(h) + Minute(m) + Second(s)

        start_st = rand(stations)
        # end station is often different but sometimes same
        end_st = rand() < 0.85 ? rand(setdiff(stations, [start_st])) : start_st

        # durations: commute-ish in rush hours, longer on weekends
        weekday = dayofweek(day) in 1:5
        if weekday
            base = (h in (8,9,17,18)) ? rand(6.0:0.5:18.0) : rand(5.0:0.5:25.0)
        else
            base = rand(8.0:0.5:35.0)
        end

        start_station[i] = start_st
        end_station[i] = end_st
        started_at[i] = started
        duration_min[i] = base
    end

    df = DataFrame(
        trip_id = trip_id,
        start_station = start_station,
        end_station = end_station,
        started_at = started_at,
        duration_min = duration_min
    )

    return df
end

trips = make_trips(2_000_000)  

# ----------------------------
# 2) Feature engineering
# ----------------------------
trips.day = Date.(trips.started_at)
trips.hour = hour.(trips.started_at)
trips.dow = dayname.(trips.day)

# Rush-hour indicator
rush_set = Set([8, 9, 17, 18])
trips.is_rush = in.(trips.hour, Ref(rush_set))

# ----------------------------
# 3) Aggregations (fast groupby)
# ----------------------------
rush_share(v) = mean(v)

by_station = combine(groupby(trips, :start_station),
    nrow => :trips,
    :duration_min => mean => :avg_duration,
    :is_rush => rush_share => :rush_share
)

sort!(by_station, :trips, rev=true)

println("\nTop stations by trip volume:")
show(first(by_station, 8), allrows=true, allcols=true)

# Hourly volume profile
by_hour = combine(groupby(trips, :hour), nrow => :trips)
sort!(by_hour, :hour)

# Day-of-week x hour "heatmap" counts
by_dow_hour = combine(groupby(trips, [:dow, :hour]), nrow => :trips)

# Make a stable ordering for the heatmap rows
dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
by_dow_hour.dow = categorical(by_dow_hour.dow; levels=dow_order, ordered=true)
sort!(by_dow_hour, [:dow, :hour])

# Pivot into a matrix: rows = DOW, cols = hour
hours_sorted = sort(unique(by_dow_hour.hour))
dows_sorted = dow_order

M = zeros(length(dows_sorted), length(hours_sorted))
for (r, d) in enumerate(dows_sorted)
    for (c, h) in enumerate(hours_sorted)
        idx = findfirst((by_dow_hour.dow .== d) .& (by_dow_hour.hour .== h))
        M[r, c] = isnothing(idx) ? 0 : by_dow_hour.trips[idx]
    end
end

# ----------------------------
# 4) Plotting
# ----------------------------
fig = Figure(size=(1100, 650))

ax1 = Axis(fig[1, 1], title="Trips by Hour (Synthetic Citi Bike)", xlabel="Hour", ylabel="Trips")
lines!(ax1, by_hour.hour, by_hour.trips)

ax2 = Axis(fig[2, 1], title="Trips Heatmap (Day of Week x Hour)", xlabel="Hour", ylabel="Day")

hm = heatmap!(ax2, M)  # just the matrix

# label axes
ax2.xticks = (1:length(hours_sorted), string.(hours_sorted))
ax2.yticks = (1:length(dows_sorted), dows_sorted)

Colorbar(fig[2, 2], hm, label="Trips")

save("rush_hour.png", fig)
println("\nSaved plot -> rush_hour.png")

# ----------------------------
# 5) “Power move”: a simple station score
# ----------------------------
# Rush Hour Score = trips * rush_share / avg_duration
by_station.rush_score = by_station.trips .* by_station.rush_share ./ by_station.avg_duration
sort!(by_station, :rush_score, rev=true)

println("\nStations ranked by Rush Hour Score:")
show(first(by_station, 8), allrows=true, allcols=true)
println()