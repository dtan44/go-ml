package neuralnetwork

import(
    "math"
)

// Activation functions
func Sigmoid(input float64) float64 {
    return 1.0/(1.0+math.Exp(input))
}

func SigmoidD(input float64) float64 {
    return input*(1-input)
}

func Relu(input float64) float64 {
    return math.Max(0, input)
}

func ReluD(input float64) float64 {
    return 0
}


func Activate(input [][]float64, weights []float64) ([]float64, bool) {
    if input == nil || weights == nil || (len(input[0])+1 != len(weights)) {
        return nil, false
    }

    n_dim := len(input)
    m_dim := len(input[0])
    res := make([]float64, n_dim)
    for i := 0; i < n_dim; i++ {
        val := 0.0
        for j := 0; j < m_dim; j++ {
            val += input[i][j]*weights[j]
        }
        // Add bias weight to result
        res[i] = val+weights[m_dim]
    }
    return res, true
}

// Cost Functions
type errorFunction func(observed [][]float64, actual [][]float64) float64

// 1/2\sum((O-A)^2)
func SimpleCost(observed [][]float64, actual [][]float64) float64 {
    var res float64 = 0

    length := len(observed)
    valueLength := len(observed[0])

    for i := 0; i < length; i++ {
        var sum float64 = 0
        for j := 0; j < valueLength; j++ {
            sum += observed[i][j] - actual[i][j]
        }
        res += math.Pow(sum, 2)
    }

    return res/2
}

// \sum(O-A)
func SimpleCostD(observed [][]float64, actual [][]float64) float64 {
    var res float64 = 0

    length := len(observed)
    valueLength := len(observed[0])

    for i := 0; i < length; i++ {
        var sum float64 = 0
        for j := 0; j < valueLength; j++ {
            sum += observed[i][j] - actual[i][j]
        }
        res += sum
    }

    return res
}






