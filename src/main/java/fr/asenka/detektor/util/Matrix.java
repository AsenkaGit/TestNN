package fr.asenka.detektor.util;

import static org.apache.commons.math3.linear.MatrixUtils.createRealMatrix;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Iterator;
import java.util.Locale;
import java.util.Objects;
import java.util.Random;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.StatUtils;

public class Matrix implements Iterable<Double> {

	private final RealMatrix matrix;
	private final int rows;
	private final int columns;

	public Matrix(RealMatrix matrix) {
		this.matrix = matrix;
		this.rows = matrix.getRowDimension();
		this.columns = matrix.getColumnDimension();
	}

	public Matrix(double[][] data) {
		this(createRealMatrix(data));
	}
	
	public Matrix(int[][] data) {
		this(data.length, data[0].length);
		
		for (int r = 0; r < rows; r++)
			for (int c = 0; c < columns; c++)
				this.matrix.setEntry(r, c, (double) data[r][c]);
	}

	public Matrix(int[] sizes) {
		this(sizes[0], sizes[1]);
	}

	public Matrix(int rows, int columns) {
		this(createRealMatrix(rows, columns));
	}

	public Matrix(int size) {
		this(size, size);
	}

	public Matrix(int rows, int columns, double value) {
		this(rows, columns);

		for (int r = 0; r < rows; r++)
			for (int c = 0; c < columns; c++)
				this.matrix.setEntry(r, c, value);
	}

	public Matrix(int size, double value) {
		this(size, size, value);
	}

	public Matrix(String content) {

		String[] contentRows = content.split(";");
		this.rows = contentRows.length;
		this.columns = contentRows[0].split(" ").length;
		this.matrix = createRealMatrix(rows, columns);

		for (int r = 0; r < rows; r++) {
			String[] numbers = contentRows[r].trim().split(" ");
			for (int c = 0; c < columns; c++)
				this.matrix.setEntry(r, c, Double.parseDouble(numbers[c]));
		}
	}
	
	public boolean isScalar() {
		return rows == 1 && columns == 1;
	}
	
	public boolean isSquare() {
		return rows == columns;
	}
	
	public boolean isRow() {
		return rows == 1;
	}
	
	public boolean isColumn() {
		return columns == 1;
	}

	public Matrix copy() {
		return new Matrix(this.matrix.copy());
	}

	public Matrix add(Matrix other) {
		MatrixUtils.checkAdditionCompatible(this.matrix, other.matrix);
		return new Matrix(this.matrix.add(other.matrix));
	}

	public Matrix add(double value) {
		return new Matrix(this.matrix.scalarAdd(value));
	}

	public Matrix subtract(Matrix other) {
		MatrixUtils.checkSubtractionCompatible(this.matrix, other.matrix);
		return new Matrix(this.matrix.subtract(other.matrix));
	}

	public Matrix subtract(double value) {
		return new Matrix(this.matrix.scalarAdd(-value));
	}

	public Matrix multiply(Matrix other) {
		MatrixUtils.checkMultiplicationCompatible(this.matrix, other.matrix);
		return new Matrix(this.matrix.multiply(other.matrix));
	}

	public Matrix multiply(double value) {
		return new Matrix(this.matrix.scalarMultiply(value));
	}

	public Matrix multiplyEachEntry(Matrix other) {

		if (rows != other.rows)
			throw new DimensionMismatchException(other.rows, rows);
		else if (columns != other.columns)
			throw new DimensionMismatchException(other.columns, columns);

		Matrix result = new Matrix(rows, columns);

		for (int r = 0; r < rows; r++)
			for (int c = 0; c < columns; c++)
				result.matrix.setEntry(r, c, this.matrix.getEntry(r, c) * other.matrix.getEntry(r, c));

		return result;
	}
	
	public Matrix divideEachEntry(Matrix other) {
	
		if (rows != other.rows)
			throw new DimensionMismatchException(other.rows, rows);
		else if (columns != other.columns)
			throw new DimensionMismatchException(other.columns, columns);

		Matrix result = new Matrix(rows, columns);

		for (int r = 0; r < rows; r++)
			for (int c = 0; c < columns; c++)
				result.matrix.setEntry(r, c, this.matrix.getEntry(r, c) / other.matrix.getEntry(r, c));

		return result;
	}

	public Matrix divide(double value) {
		return new Matrix(this.matrix.scalarMultiply(1.0 / value));
	}

	public Matrix power(int p) {
		return new Matrix(this.matrix.power(p));
	}
	
	public Matrix flatRow() {
		Matrix result = new Matrix(1, rows * columns);
		int index = 0;
		
		for (double elem : this)
			result.set(0, index++, elem);
		
		return result;
	}
	
	public Matrix flatColumn() {
		Matrix result = new Matrix(rows * columns, 1);
		int index = 0;
		
		for (double elem : this)
			result.set(index++, 0, elem);
		
		return result;
	}

	public Matrix transpose() {
		return new Matrix(this.matrix.transpose());
	}
	
	public Matrix negative() {
		return applyOnEach(x -> -x);
	}
	
	public Matrix normalize() {
		final double max = max();
		final double min = min();
		return applyOnEach(x -> (x - min) / (max - min));
	}
	

	public double trace() {
		return this.matrix.getTrace();
	}
	
	public double norm() {
		return this.matrix.getNorm();
	}

	public long size() {
		return rows * columns;
	}
	
	public int rows() {
		return rows;
	}
	
	public int columns() {
		return columns;
	}
	
	public double max() {
		return stream().max((a, b) -> Double.compare(a, b)).get();
	}
	
	public double min() {
		return stream().min((a, b) -> Double.compare(a, b)).get();
	}
	
	public Matrix maxByRow() {
		
		Matrix result = new Matrix(rows, 1);
		
		for(int r = 0; r < rows; r++)
			result.set(r, 0, getRow(r).max());
		
		return result;
	}
	
	public Matrix indexMaxByRow() {
		
		Matrix result = new Matrix(rows, 1);
		
		for(int r = 0; r < rows; r++)
			result.set(r, 0, indexMax(matrix.getRow(r)));
			
		return result;
	}
	
	public Matrix minByRow() {
		
		Matrix result = new Matrix(rows, 1);
		
		for(int r = 0; r < rows; r++)
			result.set(r, 0, getRow(r).min());
		
		return result;
	}
	
	public Matrix indexMinByRow() {
		
		Matrix result = new Matrix(rows, 1);
		
		for(int r = 0; r < rows; r++)
			result.set(r, 0, indexMin(matrix.getRow(r)));
			
		return result;
	}
	
	public Matrix maxByColumn() {
		
		Matrix result = new Matrix(1, columns);
		
		for(int c = 0; c < columns; c++)
			result.set(0, c, getColumn(c).max());
		
		return result;
	}
	
	public Matrix indexMaxByColumn() {
		
		Matrix result = new Matrix(1, columns);
		
		for(int c = 0; c < columns; c++)
			result.set(0, c, indexMax(matrix.getColumn(c)));
		
		return result;
	}
	
	public Matrix minByColumn() {
		
		Matrix result = new Matrix(1, columns);
		
		for(int c = 0; c < columns; c++)
			result.set(0, c, getColumn(c).min());
		
		return result;
	}
	
	public Matrix indexMinByColumn() {
		
		Matrix result = new Matrix(1, columns);
		
		for(int c = 0; c < columns; c++)
			result.set(0, c, indexMin(matrix.getColumn(c)));
		
		return result;
	}
	
	public Matrix concatV(Matrix other) {
		
		if (columns != other.columns)
			throw new DimensionMismatchException(other.columns, columns);
		
		Matrix result = new Matrix(rows + other.rows, columns);
		
		for (int r = 0; r < rows; r++)
			for (int c = 0; c < columns; c++)
				result.matrix.setEntry(r, c, this.matrix.getEntry(r, c));
		
		for (int r = 0; r < other.rows; r++)
			for (int c = 0; c < other.columns; c++)
				result.matrix.setEntry(rows + r, c, other.matrix.getEntry(r, c));
		
		return result;
	}
	
	public Matrix concatH(Matrix other) {
		
		if (rows != other.rows)
			throw new DimensionMismatchException(other.rows, rows);
		
		Matrix result = new Matrix(rows, columns + other.columns);
		
		for (int r = 0; r < rows; r++)
			for (int c = 0; c < columns; c++)
				result.matrix.setEntry(r, c, this.matrix.getEntry(r, c));
		
		for (int r = 0; r < other.rows; r++)
			for (int c = 0; c < other.columns; c++)
				result.matrix.setEntry(r, columns + c, other.matrix.getEntry(r, c));
		
		return result;
	}
	
	public Matrix rows(int startRow, int endRow) {
		return new Matrix(this.matrix.getSubMatrix(startRow, endRow, 0, columns - 1));
	}
	
	public Matrix columns(int startColumn, int endColumn) {
		return new Matrix(this.matrix.getSubMatrix(0, rows - 1, startColumn, endColumn));
	}
	
	public Matrix subMatrix(int startRow, int startColumn) {
		return new Matrix(this.matrix.getSubMatrix(startRow, rows - 1, startColumn, columns - 1));
	}
	
	public Matrix subMatrix(int startRow, int endRow, int startColumn, int endColumn) {
		return new Matrix(this.matrix.getSubMatrix(startRow, endRow, startColumn, endColumn));
	}
	
	public double get(int row, int column) {
		return this.matrix.getEntry(row, column);
	}
	
	public Matrix getRow(int row) {
		return new Matrix(this.matrix.getRowMatrix(row));
	}
	
	public Matrix getColumn(int column) {
		return new Matrix(this.matrix.getColumnMatrix(column));
	}
	
	public void set(int row, int column, double value) {
		this.matrix.setEntry(row, column, value);
	}
	
	public void setRow(int row, double[] array) {
		this.matrix.setRow(row, array);
	}
	
	public void setRow(int row, Matrix rowMatrix) {
		this.matrix.setRowMatrix(row, rowMatrix.matrix);
	}
	
	public void setColumn(int column, double[] array) {
		this.matrix.setColumn(column, array);
	}
	
	public void setColumn(int column, Matrix columnMatrix) {	
		this.matrix.setColumnMatrix(column, columnMatrix.matrix);
	}
	
	public void setRowWithValue(int row, double value) {
		
		for (int c = 0; c < columns; c++)
			this.matrix.setEntry(row, c, value);
	}
	
	public void setColumnWithValue(int column, double value) {
		
		for (int r = 0; r < rows; r++)
			this.matrix.setEntry(r, column, value);
	}
	
	public double[][] getRawData() {
		return this.matrix.getData();
	}
	
	public Stream<Double> stream() {
		return StreamSupport.stream(spliterator(), false);
	}
	
	public void forEachRow(Consumer<Matrix> action) {
        Objects.requireNonNull(action);
        for (int r = 0; r < rows; r++) 
            action.accept(getRow(r));
    }
	
	public void forEachColumn(Consumer<Matrix> action) {
        Objects.requireNonNull(action);
        for (int c = 0; c < columns; c++) 
            action.accept(getColumn(c));
    }
	
	public Matrix applyOnEach(Function<Double, Double> function) {
		
		Matrix result = new Matrix(rows, columns);
		
		for (int r = 0; r < rows; r++) 
			for (int c = 0; c < columns; c++)
				result.set(r, c, function.apply(matrix.getEntry(r, c)));
		
		return result;
	}

	@Override
	public Iterator<Double> iterator() {
		return new Iterator<Double>() {

			private int index = 0;
			
			@Override
			public boolean hasNext() {
				return index < rows * columns;
			}

			@Override
			public Double next() { 
				return matrix.getEntry(index / columns, index++ % columns);
			}
		};
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((matrix == null) ? 0 : matrix.hashCode());
		result = prime * result + rows;
		result = prime * result + columns;
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Matrix other = (Matrix) obj;
		if (rows != other.rows)
			return false;
		if (columns != other.columns)
			return false;
		if (!Objects.deepEquals(matrix.getData(), other.matrix.getData()))
			return false;
		return true;
	}

	@Override
	public String toString() {

		StringBuilder builder = new StringBuilder();
		NumberFormat f = NumberFormat.getInstance(Locale.getDefault());
		if (f instanceof DecimalFormat) {
			f.setMaximumFractionDigits(8);
			f.setMinimumFractionDigits(8);
		}
		final int limitedRows = rows > 10 ? 10 : rows;
		final int limitedColumns = columns > 10 ? 10 : columns;
		
		builder.append("[" + rows + ";" + columns + "]\n");
		for (int r = 0; r < limitedRows; r++) {
			builder.append('\t');
			for (int c = 0; c < limitedColumns; c++) {
				builder.append(f.format((matrix.getEntry(r, c))));
				builder.append('\t');
			}
			builder.append(limitedColumns < columns ? " ...\n" : '\n');
		}
		if (limitedRows < rows) {
			builder.append("\t.\n");
			builder.append("\t.\n");
			builder.append("\t.\n");
		}
		return builder.toString();
	}

	public static final Matrix random(int rows, int columns, double min, double max) {

		Random rng = new Random();
		Matrix result = new Matrix(rows, columns);

		for (int r = 0; r < rows; r++)
			for (int c = 0; c < columns; c++)
				result.matrix.setEntry(r, c, min + (max - min) * rng.nextDouble());
		
		return result;
	}
	
	public static final Matrix zeros(int rows, int columns) {
		return new Matrix(rows, columns, 0d);
	}
	
	public static final Matrix ones(int rows, int columns) {
		return new Matrix(rows, columns, 1d);
	}
	
	public static final Matrix sum(Matrix m) {
		return m.isRow() ? sumByRow(m) : sumByColumn(m);
	}
	
	public static final Matrix sumByColumn(Matrix m) {
		
		Matrix result = new Matrix(1, m.columns);
		
		for (int c = 0; c < m.columns; c++)
			result.set(0, c, StatUtils.sum(m.matrix.getColumn(c)));
		
		return result;
	}
	
	public static final Matrix sumByRow(Matrix m) {
		
		Matrix result = new Matrix(m.rows, 1);
		
		for (int r = 0; r < m.rows; r++)
			result.set(r, 0, StatUtils.sum(m.matrix.getRow(r)));
		
		return result;
	}
	
	public static final double sumAll(Matrix m) {
		
		double sum = 0;
		for (double elem : m)
			sum += elem;
		return sum;
	}
	
	public static final Matrix log(Matrix m) {
		return m.applyOnEach(d -> Math.log(d));
	}
	
	public static final Matrix binaryMatrix(Matrix rowIntegerMatrix, int numValues) {
		
		if (!rowIntegerMatrix.isRow()) 
			throw new DimensionMismatchException(rowIntegerMatrix.rows, 1);
		
		Matrix result = new Matrix(numValues, rowIntegerMatrix.columns);
		
		for (int r = 0; r < numValues; r++)
			for (int c = 0; c < rowIntegerMatrix.columns; c++)
				result.set(r, c, r == rowIntegerMatrix.get(0, c) ? 1d : 0d);
		
		return result;
	}
	
	private static final int indexMax(double[] array) {
		double largest = array[0];
		int index = 0;
		
		for (int i = 1; i < array.length; i++)
			if (array[i] >= largest) {
				largest = array[i];
				index = i;
			}
		return index;
	}
	
	private static final int indexMin(double[] array) {
		double lowest = array[0];
		int index = 0;
		
		for (int i = 1; i < array.length; i++)
			if (array[i] < lowest) {
				lowest = array[i];
				index = i;
			}
		return index;
	}
}
