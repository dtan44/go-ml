package neural

import(
    "math/rand"
)

type Neuron struct {
    weights []float64
    error float64
}

func (neuron *Neuron) Init(size int)  {
    // add bias
    size += 1

    weights := make([]float64, size)
    for i := 0; i < size; i++ {
        weights[i] = rand.Float64()
    }
    neuron.weights = weights
}
