
## Helper functions to collect data when sampling
"""
Initialises a named tuple containing Vectors with undefined values.
Used with samplers to store results. State just provides an "example" state
from which to infer types of vectors in the array. Names of the named tuple
are the symbols provided.

"""
function init_data_tuple(
    state::AbstractSamplerState,
    collect_data::Vector{Symbol},
    n_steps::Integer)

    names = collect_data
    values = Vector{Array}(undef, length(names))

    for (i, symbol) in enumerate(collect_data)
        values[i] = Vector{typeof(getproperty(state, symbol))}(undef, n_steps)
    end
    (;zip(names, values)...)
end


"""
Add data to the data tuple.
"""
function add_state!(
    data::NamedTuple, state::AbstractSamplerState, idx::Integer)
    for symbol in keys(data)
        field = getproperty(state, symbol)
        data[symbol][idx] = field
    end
    data
end


"""
Loop through named tuple and call stack_arrays on any vector whose
elements are an array. Used at end of samplers.
"""
function simplify_data(data::NamedTuple)
    symbols = keys(data)
    new_values = Vector{Array}(undef, length(symbols))

    for (i, x) in enumerate(data)
        if x[1] isa Array
            new_values[i] = stack_arrays(x)
        else
            new_values[i] = x
        end
    end
    (;zip(symbols, new_values)...)
end
