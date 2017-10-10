package neural

type Network struct {
    hiddenLayers []Layer
    outputLayer Layer
    input [][]float64
}

func (network *Network) Init(n_features, depth, width, n_outputs int) {
    hiddenLayers := make([]Layer, depth)
    for i := 0; i < depth; i++ {
        hiddenLayers[i] = Layer{}

        if i == 0 {
            hiddenLayers[i].Init(width, n_features)
        } else {
            hiddenLayers[i].Init(width, width)
        }
    }
    outputLayer := Layer{}
    outputLayer.Init(n_outputs, width)

    network.hiddenLayers = hiddenLayers
    network.outputLayer = outputLayer
}

func (network *Network) Data(input [][]float64) {
    network.input = input
}
