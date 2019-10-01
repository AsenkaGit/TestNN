package fr.asenka.detektor.util;

import static org.apache.commons.math3.linear.MatrixUtils.createRealMatrix;

import java.util.Random;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public class Matrix {

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

	public Matrix copy() {
		return new Matrix(this.matrix.copy());
	}

	public Matrix add(Matrix other) {
		MatrixUtils.checkSubtractionCompatible(this.matrix, other.matrix);
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

		if (rows != other.getRows() || columns != other.getColumns())
			throw new IllegalArgumentException(
					" this.rows=" + rows + 
					" this.columns=" + columns + 
					" other.rows="+ other.getRows() + 
					" other.columns=" + other.getColumns());

		Matrix result = new Matrix(rows, columns);

		for (int r = 0; r < rows; r++)
			for (int c = 0; c < columns; c++)
				result.matrix.setEntry(r, c, this.matrix.getEntry(r, c) * other.matrix.getEntry(r, c));

		return result;
	}
	
	public Matrix divideEachEntry(Matrix other) {

		if (rows != other.getRows() || columns != other.getColumns())
			throw new IllegalArgumentException(
					" this.rows=" + rows + 
					" this.columns=" + columns + 
					" other.rows="+ other.getRows() + 
					" other.columns=" + other.getColumns());

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

	public Matrix transpose() {
		return new Matrix(this.matrix.transpose());
	}

	public double trace() {
		return this.matrix.getTrace();
	}

	public int getRows() {
		return matrix.getRowDimension();
	}

	public int getColumns() {
		return matrix.getColumnDimension();
	}

	public int[] sizes() {
		return new int[] { matrix.getRowDimension(), matrix.getColumnDimension() };
	}
	
	public Matrix concatVerticaly(Matrix other) {
		
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
	
	public Matrix concatHorizontaly(Matrix other) {
		
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
	
	public Matrix getSubMatrix(int startRow, int startColumn) {
		return new Matrix(this.matrix.getSubMatrix(startRow, rows - 1, startColumn, columns - 1));
	}
	
	public Matrix getSubMatrix(int startRow, int endRow, int startColumn, int endColumn) {
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

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + columns;
		result = prime * result + ((matrix == null) ? 0 : matrix.hashCode());
		result = prime * result + rows;
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
		if (columns != other.columns)
			return false;
		if (matrix == null) {
			if (other.matrix != null)
				return false;
		} else if (!matrix.equals(other.matrix))
			return false;
		if (rows != other.rows)
			return false;
		return true;
	}

	@Override
	public String toString() {

		StringBuilder builder = new StringBuilder();

		builder.append("[" + rows + ";" + columns + "]\n");
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < columns; c++) {
				builder.append('\t');
				builder.append(matrix.getEntry(r, c));
				builder.append('\t');
			}
			builder.append('\n');
		}
		return builder.toString();
	}

	public static Matrix random(int rows, int columns, double min, double max) {

		Random rng = new Random();
		Matrix result = new Matrix(rows, columns);

		for (int r = 0; r < rows; r++)
			for (int c = 0; c < columns; c++)
				result.matrix.setEntry(r, c, min + (max - min) * rng.nextDouble());

		return result;
	}
	
	public static Matrix zeros(int rows, int columns) {
		return new Matrix(rows, columns, 0d);
	}
	
	public static Matrix ones(int rows, int columns) {
		return new Matrix(rows, columns, 1d);
	}
}
