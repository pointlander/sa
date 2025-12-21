// Copyright 2025 The SA Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/zip"
	"bytes"
	"compress/bzip2"
	"embed"
	"encoding/csv"
	"flag"
	"fmt"
	"image/color"
	"io"
	"math"
	"math/rand"
	"sort"
	"strconv"
	"strings"

	"github.com/pointlander/gradient/tf64"
	"github.com/pointlander/sa/kmeans"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

const (
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.8
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.89
	// Eta is the learning rate
	Eta = 1.0e-3
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

//go:embed iris.zip
var Iris embed.FS

// Fisher is the fisher iris data
type Fisher struct {
	Measures  []float64
	Embedding []float64
	Label     string
	L         byte
	Cluster   int
	Index     int
}

// Labels maps iris labels to ints
var Labels = map[string]int{
	"Iris-setosa":     0,
	"Iris-versicolor": 1,
	"Iris-virginica":  2,
	"gen":             3,
}

// Inverse is the labels inverse map
var Inverse = [4]string{
	"Iris-setosa",
	"Iris-versicolor",
	"Iris-virginica",
	"gen",
}

// Load loads the iris data set
func Load() []Fisher {
	file, err := Iris.Open("iris.zip")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		panic(err)
	}

	fisher := make([]Fisher, 0, 8)
	reader, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		panic(err)
	}
	for _, f := range reader.File {
		if f.Name == "iris.data" {
			iris, err := f.Open()
			if err != nil {
				panic(err)
			}
			reader := csv.NewReader(iris)
			data, err := reader.ReadAll()
			if err != nil {
				panic(err)
			}
			for i, item := range data {
				record := Fisher{
					Measures: make([]float64, 4),
					Label:    item[4],
					Index:    i,
				}
				for ii := range item[:4] {
					f, err := strconv.ParseFloat(item[ii], 64)
					if err != nil {
						panic(err)
					}
					record.Measures[ii] = f
				}
				fisher = append(fisher, record)
			}
			iris.Close()
		}
	}
	return fisher
}

//go:embed books/*
var Books embed.FS

// Book is a book
type Book struct {
	Name string
	Text []byte
}

// LoadBooks loads books
func LoadBooks() []Book {
	books := []Book{
		{Name: "pg74.txt.bz2"},
		{Name: "10.txt.utf-8.bz2"},
		{Name: "76.txt.utf-8.bz2"},
		{Name: "84.txt.utf-8.bz2"},
		{Name: "100.txt.utf-8.bz2"},
		{Name: "1837.txt.utf-8.bz2"},
		{Name: "2701.txt.utf-8.bz2"},
		{Name: "3176.txt.utf-8.bz2"},
	}
	load := func(book string) []byte {
		file, err := Books.Open(book)
		if err != nil {
			panic(err)
		}
		defer file.Close()
		breader := bzip2.NewReader(file)
		data, err := io.ReadAll(breader)
		if err != nil {
			panic(err)
		}
		return data
	}
	for i := range books {
		books[i].Text = load(fmt.Sprintf("books/%s", books[i].Name))
	}
	return books
}

// LearnEmbedding learns the embedding
func LearnEmbedding(iris []Fisher, average []float64, size, width int) []Fisher {
	rng := rand.New(rand.NewSource(1))
	others := tf64.NewSet()
	length := len(iris)
	if *FlagGen {
		length++
	}
	cp := make([]Fisher, length)
	copy(cp, iris)
	if *FlagGen {
		cp[len(iris)].Label = "gen"
		cp[len(iris)].Index = len(iris)
	}
	others.Add("x", size, len(cp))
	x := others.ByName["x"]
	for _, row := range iris {
		x.X = append(x.X, row.Measures...)
	}
	if *FlagGen {
		w := x
		for i := range size {
			w.X = append(w.X, rng.Float64()*average[i])
		}
		w.States = make([][]float64, StateTotal)
		for ii := range w.States {
			w.States[ii] = make([]float64, len(w.X))
		}
	}

	set := tf64.NewSet()
	set.Add("i", width, len(cp))

	for ii := range set.Weights {
		w := set.Weights[ii]
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:cap(w.X)]
			w.States = make([][]float64, StateTotal)
			for ii := range w.States {
				w.States[ii] = make([]float64, len(w.X))
			}
			continue
		}
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for range cap(w.X) {
			w.X = append(w.X, rng.NormFloat64()*factor*.01)
		}
		w.States = make([][]float64, StateTotal)
		for ii := range w.States {
			w.States[ii] = make([]float64, len(w.X))
		}
	}

	drop := .3
	dropout := map[string]interface{}{
		"rng":  rng,
		"drop": &drop,
	}

	sa := tf64.T(tf64.Mul(tf64.Dropout(tf64.Square(set.Get("i")), dropout), tf64.T(others.Get("x"))))
	loss := tf64.Avg(tf64.Quadratic(others.Get("x"), sa))

	for iteration := range 2 * 1024 {
		pow := func(x float64) float64 {
			y := math.Pow(x, float64(iteration+1))
			if math.IsNaN(y) || math.IsInf(y, 0) {
				return 0
			}
			return y
		}

		set.Zero()
		others.Zero()
		l := tf64.Gradient(loss).X[0]
		if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
			fmt.Println(iteration, l)
			return nil
		}

		norm := 0.0
		for _, p := range set.Weights {
			for _, d := range p.D {
				norm += d * d
			}
		}
		norm = math.Sqrt(norm)
		b1, b2 := pow(B1), pow(B2)
		scaling := 1.0
		if norm > 1 {
			scaling = 1 / norm
		}
		for _, w := range set.Weights {
			for ii, d := range w.D {
				g := d * scaling
				m := B1*w.States[StateM][ii] + (1-B1)*g
				v := B2*w.States[StateV][ii] + (1-B2)*g*g
				w.States[StateM][ii] = m
				w.States[StateV][ii] = v
				mhat := m / (1 - b1)
				vhat := v / (1 - b2)
				if vhat < 0 {
					vhat = 0
				}
				w.X[ii] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
			}
		}
		if *FlagGen {
			w := x
			offset := len(iris) * size
			D := w.D[offset:]
			for ii, d := range D {
				g := d * scaling
				m := B1*w.States[StateM][offset+ii] + (1-B1)*g
				v := B2*w.States[StateV][offset+ii] + (1-B2)*g*g
				w.States[StateM][offset+ii] = m
				w.States[StateV][offset+ii] = v
				mhat := m / (1 - b1)
				vhat := v / (1 - b2)
				if vhat < 0 {
					vhat = 0
				}
				w.X[offset+ii] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
			}
		}
		fmt.Println(l)
	}

	if *FlagGen {
		for iteration := range 2 * 1024 {
			pow := func(x float64) float64 {
				y := math.Pow(x, float64(iteration+1))
				if math.IsNaN(y) || math.IsInf(y, 0) {
					return 0
				}
				return y
			}

			set.Zero()
			others.Zero()
			l := tf64.Gradient(loss).X[0]
			if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
				fmt.Println(iteration, l)
				return nil
			}

			norm := 0.0
			for _, p := range set.Weights {
				for _, d := range p.D {
					norm += d * d
				}
			}
			norm = math.Sqrt(norm)
			b1, b2 := pow(B1), pow(B2)
			scaling := 1.0
			if norm > 1 {
				scaling = 1 / norm
			}
			{
				w := x
				offset := len(iris) * size
				D := w.D[offset:]
				for ii, d := range D {
					g := d * scaling
					m := B1*w.States[StateM][offset+ii] + (1-B1)*g
					v := B2*w.States[StateV][offset+ii] + (1-B2)*g*g
					w.States[StateM][offset+ii] = m
					w.States[StateV][offset+ii] = v
					mhat := m / (1 - b1)
					vhat := v / (1 - b2)
					if vhat < 0 {
						vhat = 0
					}
					w.X[offset+ii] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
				}
			}
			fmt.Println(l)
		}
	}

	meta := make([][]float64, len(cp))
	for i := range meta {
		meta[i] = make([]float64, len(cp))
	}
	const k = 3

	{
		y := set.ByName["i"]
		vectors := make([][]float64, len(cp))
		for i := range vectors {
			row := make([]float64, width)
			for ii := range row {
				row[ii] = y.X[i*width+ii]
			}
			vectors[i] = row
		}
		for i := 0; i < 33; i++ {
			clusters, _, err := kmeans.Kmeans(int64(i+1), vectors, k, kmeans.SquaredEuclideanDistance, -1)
			if err != nil {
				panic(err)
			}
			for i := 0; i < len(meta); i++ {
				target := clusters[i]
				for j, v := range clusters {
					if v == target {
						meta[i][j]++
					}
				}
			}
		}
	}
	clusters, _, err := kmeans.Kmeans(1, meta, 3, kmeans.SquaredEuclideanDistance, -1)
	if err != nil {
		panic(err)
	}
	for i := range clusters {
		cp[i].Cluster = clusters[i]
	}
	for _, value := range x.X[len(iris)*size:] {
		cp[len(iris)].Measures = append(cp[len(iris)].Measures, value)
	}
	I := set.ByName["i"]
	for i := range cp {
		cp[i].Embedding = I.X[i*width : (i+1)*width]
	}
	sort.Slice(cp, func(i, j int) bool {
		return cp[i].Cluster < cp[j].Cluster
	})
	return cp
}

// LearnEmbeddingAlpha learns the embedding
func LearnEmbeddingAlpha(iris []Fisher, size, width int) []Fisher {
	rng := rand.New(rand.NewSource(1))
	others := tf64.NewSet()
	length := len(iris)
	if *FlagGen {
		length++
	}
	cp := make([]Fisher, length)
	copy(cp, iris)
	others.Add("x", size, len(cp))
	x := others.ByName["x"]
	for _, row := range iris {
		x.X = append(x.X, row.Measures...)
	}

	set := tf64.NewSet()
	set.Add("i", width, len(cp))
	set.Add("w0", size, size)
	set.Add("b0", size)
	set.Add("w1", 2*size, size)
	set.Add("b1", size)

	for ii := range set.Weights {
		w := set.Weights[ii]
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:cap(w.X)]
			w.States = make([][]float64, StateTotal)
			for ii := range w.States {
				w.States[ii] = make([]float64, len(w.X))
			}
			continue
		}
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for range cap(w.X) {
			w.X = append(w.X, rng.NormFloat64()*factor)
		}
		w.States = make([][]float64, StateTotal)
		for ii := range w.States {
			w.States[ii] = make([]float64, len(w.X))
		}
	}

	drop := .3
	dropout := map[string]interface{}{
		"rng":  rng,
		"drop": &drop,
	}

	l0 := tf64.Everett(tf64.Add(tf64.Mul(set.Get("w0"), others.Get("x")), set.Get("b0")))
	l1 := tf64.Add(tf64.Mul(set.Get("w1"), l0), set.Get("b1"))
	sa := tf64.T(tf64.Mul(tf64.Dropout(tf64.Square(set.Get("i")), dropout), tf64.T(l1)))
	loss := tf64.Avg(tf64.Quadratic(l1, sa))

	for iteration := range 256 {
		pow := func(x float64) float64 {
			y := math.Pow(x, float64(iteration+1))
			if math.IsNaN(y) || math.IsInf(y, 0) {
				return 0
			}
			return y
		}

		set.Zero()
		others.Zero()
		l := tf64.Gradient(loss).X[0]
		if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
			fmt.Println(iteration, l)
			return nil
		}

		norm := 0.0
		for _, p := range set.Weights {
			for _, d := range p.D {
				norm += d * d
			}
		}
		norm = math.Sqrt(norm)
		b1, b2 := pow(B1), pow(B2)
		scaling := 1.0
		if norm > 1 {
			scaling = 1 / norm
		}
		for _, w := range set.Weights {
			for ii, d := range w.D {
				g := d * scaling
				m := B1*w.States[StateM][ii] + (1-B1)*g
				v := B2*w.States[StateV][ii] + (1-B2)*g*g
				w.States[StateM][ii] = m
				w.States[StateV][ii] = v
				mhat := m / (1 - b1)
				vhat := v / (1 - b2)
				if vhat < 0 {
					vhat = 0
				}
				w.X[ii] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
			}
		}
		fmt.Println(iteration, l)
	}

	meta := make([][]float64, len(cp))
	for i := range meta {
		meta[i] = make([]float64, len(cp))
	}
	const k = 3

	{
		y := set.ByName["i"]
		vectors := make([][]float64, len(cp))
		for i := range vectors {
			row := make([]float64, width)
			for ii := range row {
				row[ii] = y.X[i*width+ii]
			}
			vectors[i] = row
		}
		for i := 0; i < 33; i++ {
			clusters, _, err := kmeans.Kmeans(int64(i+1), vectors, k, kmeans.SquaredEuclideanDistance, -1)
			if err != nil {
				panic(err)
			}
			for i := 0; i < len(meta); i++ {
				target := clusters[i]
				for j, v := range clusters {
					if v == target {
						meta[i][j]++
					}
				}
			}
		}
	}
	clusters, _, err := kmeans.Kmeans(1, meta, 3, kmeans.SquaredEuclideanDistance, -1)
	if err != nil {
		panic(err)
	}
	for i := range clusters {
		cp[i].Cluster = clusters[i]
	}
	I := set.ByName["i"]
	for i := range cp {
		cp[i].Embedding = I.X[i*width : (i+1)*width]
	}
	/*sort.Slice(cp, func(i, j int) bool {
		return cp[i].Cluster < cp[j].Cluster
	})*/
	return cp
}

// Stats are the statistics for a model
type Stats struct {
	Min float64
	Max float64
}

// Network is a feedforward neural network
type Network struct {
	Rng    *rand.Rand
	Width  int
	Stats  []Stats
	Set    tf64.Set
	Others tf64.Set
	L0     tf64.Meta
	L1     tf64.Meta
	Loss   tf64.Meta
}

// NewNetwork creates a network
func NewNetwork(width int) Network {
	rng := rand.New(rand.NewSource(1))
	stats := make([]Stats, width)
	for i := range stats {
		stats[i].Min = math.MaxFloat64
		stats[i].Max = -math.MaxFloat64
	}
	set := tf64.NewSet()
	set.Add("w0", width, width)
	set.Add("b0", width)
	set.Add("w1", 2*width, 4)
	set.Add("b1", 4, 1)
	for i := range set.Weights {
		w := set.Weights[i]
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:cap(w.X)]
			w.States = make([][]float64, StateTotal)
			for ii := range w.States {
				w.States[ii] = make([]float64, len(w.X))
			}
			continue
		}
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for range cap(w.X) {
			w.X = append(w.X, rng.NormFloat64()*factor)
		}
		w.States = make([][]float64, StateTotal)
		for ii := range w.States {
			w.States[ii] = make([]float64, len(w.X))
		}
	}

	others := tf64.NewSet()
	others.Add("input", width)
	others.Add("output", 4)
	input := others.ByName["input"].X
	others.ByName["input"].X = input[:cap(input)]
	output := others.ByName["output"].X
	others.ByName["output"].X = output[:cap(output)]

	l0 := tf64.Everett(tf64.Add(tf64.Mul(set.Get("w0"), others.Get("input")), set.Get("b0")))
	l1 := tf64.Add(tf64.Mul(set.Get("w1"), l0), set.Get("b1"))
	loss := tf64.Quadratic(others.Get("output"), l1)

	return Network{
		Rng:    rng,
		Width:  width,
		Stats:  stats,
		Set:    set,
		Others: others,
		L0:     l0,
		L1:     l1,
		Loss:   loss,
	}
}

// Learn learns with a network
func (n *Network) Learn(data []Fisher) {
	for i := range data {
		for ii := range n.Stats {
			value := data[i].Embedding[ii]
			if value > n.Stats[ii].Max {
				n.Stats[ii].Max = value
			}
			if value < n.Stats[ii].Min {
				n.Stats[ii].Min = value
			}
		}
	}
	for iteration := range 2 * 1024 {
		index := n.Rng.Intn(len(data))
		copy(n.Others.ByName["input"].X, data[index].Embedding)
		copy(n.Others.ByName["output"].X, data[index].Measures)
		pow := func(x float64) float64 {
			y := math.Pow(x, float64(iteration+1))
			if math.IsNaN(y) || math.IsInf(y, 0) {
				return 0
			}
			return y
		}

		n.Set.Zero()
		n.Others.Zero()
		l := tf64.Gradient(n.Loss).X[0]
		if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
			fmt.Println(iteration, l)
		}

		norm := 0.0
		for _, p := range n.Set.Weights {
			for _, d := range p.D {
				norm += d * d
			}
		}
		norm = math.Sqrt(norm)
		b1, b2 := pow(B1), pow(B2)
		scaling := 1.0
		if norm > 1 {
			scaling = 1 / norm
		}
		for _, w := range n.Set.Weights {
			for ii, d := range w.D {
				g := d * scaling
				m := B1*w.States[StateM][ii] + (1-B1)*g
				v := B2*w.States[StateV][ii] + (1-B2)*g*g
				w.States[StateM][ii] = m
				w.States[StateV][ii] = v
				mhat := m / (1 - b1)
				vhat := v / (1 - b2)
				if vhat < 0 {
					vhat = 0
				}
				w.X[ii] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
			}
		}
		fmt.Println(l)
	}
}

var (
	// FlagGen generation mode
	FlagGen = flag.Bool("gen", false, "generation mode")
	// FlagBook book mode
	FlagBook = flag.Bool("book", false, "book mode")
	// FlagPlot plots the embeddings
	FlagPlot = flag.Bool("plot", false, "plot the embeddings")
)

func main() {
	flag.Parse()

	if *FlagBook {
		const (
			Eta = 1.0e-3
		)
		books := LoadBooks()
		book := make([]Fisher, 0, 8)
		offset := 3 * 1024
		input := []byte{}
		for i, symbol := range books[1].Text[offset : offset+1024] {
			b := Fisher{
				Measures: make([]float64, 256),
				L:        symbol,
				Index:    i,
			}
			b.Measures[symbol] = 1
			input = append(input, symbol)
			book = append(book, b)
		}
		fmt.Println(string(input))
		width := 5
		if *FlagPlot {
			width = 2
		}
		cp := LearnEmbeddingAlpha(book, 256, width)

		if *FlagPlot {
			points := make(plotter.XYs, 0, 8)
			for _, point := range cp {
				embedding := point.Embedding
				points = append(points, plotter.XY{X: embedding[0], Y: embedding[1]})
				fmt.Println(embedding[0], embedding[1])
			}

			p := plot.New()

			p.Title.Text = "x vs y"
			p.X.Label.Text = "x"
			p.Y.Label.Text = "y"

			scatter, err := plotter.NewScatter(points)
			if err != nil {
				panic(err)
			}
			scatter.GlyphStyle.Radius = vg.Length(1)
			scatter.GlyphStyle.Shape = draw.CircleGlyph{}
			p.Add(scatter)

			err = p.Save(8*vg.Inch, 8*vg.Inch, "clusters_text.png")
			if err != nil {
				panic(err)
			}
		}

		dot := func(a, b []float64) float64 {
			x := 0.0
			for i, value := range a {
				x += value * b[i]
			}
			return x
		}

		cs := func(a, b []float64) float64 {
			ab := dot(a, b)
			aa := dot(a, a)
			bb := dot(b, b)
			if aa <= 0 {
				return 0
			}
			if bb <= 0 {
				return 0
			}
			return ab / (math.Sqrt(aa) * math.Sqrt(bb))
		}

		rng := rand.New(rand.NewSource(1))
		type Markov [2]byte
		type Bucket struct {
			Entries []Fisher
		}
		var markov Markov
		model := make(map[Markov]*Bucket)
		for _, entry := range cp {
			bucket := model[markov]
			if bucket == nil {
				bucket = &Bucket{}
			}
			bucket.Entries = append(bucket.Entries, entry)
			model[markov] = bucket
			markov[0], markov[1] = markov[1], entry.L
		}
		symbols := make([]byte, 0, 33)
		current := cp[len(cp)-1].Embedding
		for range 33 {
			bucket := model[markov]
			d := make([]float64, len(bucket.Entries))
			sum := 0.0
			for i, entry := range bucket.Entries {
				x := cs(current, entry.Embedding)
				d[i] = x
				sum += x
			}
			total, selected, index := 0.0, rng.Float64(), 0
			for i, value := range d {
				total += value / sum
				if selected < total {
					index = i
					break
				}
			}
			symbols = append(symbols, bucket.Entries[index].L)
			current = bucket.Entries[index].Embedding
			markov[0], markov[1] = markov[1], bucket.Entries[index].L
		}
		fmt.Println("`" + string(input) + "`")
		fmt.Println("`" + string(symbols) + "`")

		/*search := func() ([]byte, float64) {
			result, cost := []byte{}, 0.0
			current := cp[len(cp)-1]
			for range 33 {
				d := make([]float64, len(cp))
				for i := range cp {
					d[i] = cs(current.Embedding, cp[i].Embedding)
				}
				sum := 0.0
				for _, value := range d {
					sum += value
				}
				total, selected, index := 0.0, rng.Float64(), 0
				for i := range d {
					total += d[i] / sum
					if selected < total {
						index = i
						break
					}
				}
				for i, value := range cp[index].Embedding {
					current.Embedding[i] += value
				}
				result = append(result, cp[index].L)
				cost += d[index] / sum
			}
			return result, cost
		}
		fmt.Println("`" + string(input) + "`")
		type Result struct {
			Symbols []byte
			Cost    float64
		}
		results := make([]Result, 0, 8)
		for range 8 * 1024 {
			symbols, cost := search()
			results = append(results, Result{
				Symbols: symbols,
				Cost:    cost,
			})
		}
		sort.Slice(results, func(i, j int) bool {
			return results[i].Cost > results[j].Cost
		})
		fmt.Println("`" + string(results[0].Symbols) + "`")*/

		/*rng := rand.New(rand.NewSource(1))

		others := make([]tf64.Set, len(book))
		for i := range book {
			others[i] = tf64.NewSet()
			others[i].Add("input", 2)
			others[i].Add("output", 256)
			input := others[i].ByName["input"]
			input.X = append(input.X, book[i].Embedding...)
			output := others[i].ByName["output"]
			output.X = append(output.X, book[i].Measures...)
		}

		initial := tf64.NewSet()
		initial.Add("initial", 8)
		init := initial.ByName["initial"]
		init.X = init.X[:cap(init.X)]

		set := tf64.NewSet()
		set.Add("w0", 10, 80)
		set.Add("b0", 80)
		set.Add("w1", 160, 8+256)
		set.Add("b1", 8+256)
		for i := range set.Weights {
			w := set.Weights[i]
			if strings.HasPrefix(w.N, "b") {
				w.X = w.X[:cap(w.X)]
				w.States = make([][]float64, StateTotal)
				for ii := range w.States {
					w.States[ii] = make([]float64, len(w.X))
				}
				continue
			}
			factor := math.Sqrt(2.0 / float64(w.S[0]))
			for range cap(w.X) {
				w.X = append(w.X, rng.NormFloat64()*factor)
			}
			w.States = make([][]float64, StateTotal)
			for ii := range w.States {
				w.States[ii] = make([]float64, len(w.X))
			}
		}

		l0 := tf64.Everett(tf64.Add(tf64.Mul(set.Get("w0"), tf64.Concat(others[0].Get("input"), initial.Get("initial"))), set.Get("b0")))
		l1 := tf64.Add(tf64.Mul(set.Get("w1"), l0), set.Get("b1"))
		begin, end := 8, 8+256
		options := map[string]interface{}{
			"begin": &begin,
			"end":   &end,
		}
		loss := tf64.Quadratic(others[0].Get("output"), tf64.Slice(l1, options))

		begin2, end2 := 0, 8
		options2 := map[string]interface{}{
			"begin": &begin2,
			"end":   &end2,
		}
		for i := range book[1:] {
			l0 = tf64.Everett(tf64.Add(tf64.Mul(set.Get("w0"), tf64.Concat(others[i+1].Get("input"), tf64.Slice(l1, options2))), set.Get("b0")))
			l1 = tf64.Add(tf64.Mul(set.Get("w1"), l0), set.Get("b1"))
			loss = tf64.Add(tf64.Quadratic(others[i+1].Get("output"), tf64.Slice(l1, options)), loss)
		}

		for iteration := range 512 {
			pow := func(x float64) float64 {
				y := math.Pow(x, float64(iteration+1))
				if math.IsNaN(y) || math.IsInf(y, 0) {
					return 0
				}
				return y
			}

			set.Zero()
			initial.Zero()
			for i := range others {
				others[i].Zero()
			}
			l := tf64.Gradient(loss).X[0]
			if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
				fmt.Println(iteration, l)
				return
			}

			norm := 0.0
			for _, p := range set.Weights {
				for _, d := range p.D {
					norm += d * d
				}
			}
			norm = math.Sqrt(norm)
			b1, b2 := pow(B1), pow(B2)
			scaling := 1.0
			if norm > 1 {
				scaling = 1 / norm
			}
			for _, w := range set.Weights {
				for ii, d := range w.D {
					g := d * scaling
					m := B1*w.States[StateM][ii] + (1-B1)*g
					v := B2*w.States[StateV][ii] + (1-B2)*g*g
					w.States[StateM][ii] = m
					w.States[StateV][ii] = v
					mhat := m / (1 - b1)
					vhat := v / (1 - b2)
					if vhat < 0 {
						vhat = 0
					}
					w.X[ii] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
				}
			}
			fmt.Println(iteration, l)
		}*/

		return
	}

	iris := Load()
	average := make([]float64, 4)
	for _, row := range iris {
		for i, value := range row.Measures {
			average[i] += value
		}
	}
	for i, value := range average {
		average[i] = value / float64(len(iris))
	}

	cp := LearnEmbedding(iris, average, 4, 2)
	acc := make(map[string][4]int)
	for i := range cp {
		fmt.Println(cp[i].Cluster, cp[i].Label)
		counts := acc[cp[i].Label]
		counts[cp[i].Cluster]++
		acc[cp[i].Label] = counts
	}

	cp5 := LearnEmbedding(iris, average, 4, 5)
	acc5 := make(map[string][4]int)
	for i := range cp5 {
		fmt.Println(cp5[i].Cluster, cp5[i].Label)
		counts := acc5[cp5[i].Label]
		counts[cp5[i].Cluster]++
		acc5[cp5[i].Label] = counts
	}

	indexes := make([]*Fisher, len(cp))
	for i := range cp {
		indexes[cp[i].Index] = &cp[i]
	}

	points := make([]plotter.XYs, 4)
	networks, sets := make([]Network, 3), make(map[int][]Fisher, 3)
	for i := range networks {
		networks[i] = NewNetwork(5)
	}
	for i := range len(cp5) {
		embedding := indexes[cp5[i].Index].Embedding
		points[cp5[i].Cluster] = append(points[cp5[i].Cluster], plotter.XY{X: embedding[0], Y: embedding[1]})
		set := sets[cp5[i].Cluster]
		set = append(set, cp5[i])
		sets[cp5[i].Cluster] = set
	}
	p := plot.New()

	p.Title.Text = "x vs y"
	p.X.Label.Text = "x"
	p.Y.Label.Text = "y"

	colors := make([]color.RGBA, len(points))
	colors[0] = color.RGBA{R: 255, A: 255}
	colors[1] = color.RGBA{G: 255, A: 255}
	colors[2] = color.RGBA{B: 255, A: 255}
	colors[3] = color.RGBA{R: 255, G: 255, A: 255}

	for i := range points {
		scatter, err := plotter.NewScatter(points[i])
		if err != nil {
			panic(err)
		}
		scatter.GlyphStyle.Radius = vg.Length(1)
		scatter.GlyphStyle.Shape = draw.CircleGlyph{}
		scatter.GlyphStyle.Color = colors[i]
		p.Add(scatter)
	}

	err := p.Save(8*vg.Inch, 8*vg.Inch, "clusters.png")
	if err != nil {
		panic(err)
	}

	for i := range networks {
		fmt.Println("--------------------------------------------------")
		networks[i].Learn(sets[i])
	}

	for i, v := range acc {
		fmt.Println(i, v)
	}
	fmt.Println()
	for i, v := range acc5 {
		fmt.Println(i, v)
	}

	fmt.Println(average)
	if *FlagGen {
		for _, value := range cp {
			if value.Label == "gen" {
				fmt.Println(value.Measures)
			}
		}
		for _, value := range cp5 {
			if value.Label == "gen" {
				fmt.Println(value.Measures)
			}
		}
	}

	inputs := make([]float64, 5)
	for i := range inputs {
		inputs[i] = (networks[0].Stats[i].Max-networks[0].Stats[i].Min)*networks[0].Rng.Float64() + networks[0].Stats[i].Min
	}
	copy(networks[0].Others.ByName["input"].X, inputs)
	networks[0].L1(func(a *tf64.V) bool {
		fmt.Println(a.X)
		return true
	})
}
