module.exports = {
  entry: './demo1.ts',
  output: {
    filename: '../static/bundle.js'
  },
  resolve: {
    // Add `.ts` and `.tsx` as a resolvable extension.
    extensions: ['', '.webpack.js', '.web.js', '.ts', '.js']
  },
  module: {
    loaders: [
      // all files with a `.ts` or `.tsx` extension will be handled by `ts-loader`
        { test: /\.ts?$/, loader: 'ts-loader' },
    ]
  },
    devtool: "source-map"
}
