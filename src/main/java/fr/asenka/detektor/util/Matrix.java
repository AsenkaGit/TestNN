package fr.asenka.detektor.util;

import static org.apache.commons.math3.linear.MatrixUtils.createRealMatrix;

import java.util.Random;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public class Matrix {

	private RealMatrix matrix;

	public Matrix(RealMatrix matrix) {
		this.matrix = matrix;
	}

	public Matrix(double[][] data) {
		this.matrix = createRealMatrix(data);
	}

	public Matrix(int[] sizes) {
		this(sizes[0], sizes[1]);
	}

	public Matrix(int rows, int columns) {
		this.matrix = createRealMatrix(rows, columns);
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
		int rows = contentRows.length;
		int columns = contentRows[0].split(" ").length;

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

		int rows = matrix.getRowDimension();
		int columns = matrix.getColumnDimension();

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

		int rows = matrix.getRowDimension();
		int columns = matrix.getColumnDimension();

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
	
	public Matrix concatV(Matrix other) {
		
		int rows = matrix.getRowDimension();
		int columns = matrix.getColumnDimension();
		int otherRows = other.matrix.getRowDimension();
		int otherColumns = other.matrix.getColumnDimension();
		
		if (columns != otherColumns)
			throw new DimensionMismatchException(otherColumns, columns);
		
		Matrix result = new Matrix(rows + otherRows, columns);
		
		for (int r = 0; r < rows; r++)
			for (int c = 0; c < columns; c++)
				result.matrix.setEntry(r, c, this.matrix.getEntry(r, c));
		
		for (int r = 0; r < otherRows; r++)
			for (int c = 0; c < otherColumns; c++)
				result.matrix.setEntry(rows + r, c, other.matrix.getEntry(r, c));
		
		return result;
	}
	
	public Matrix concatH(Matrix other) {
		
		int rows = matrix.getRowDimension();
		int columns = matrix.getColumnDimension();
		int otherRows = other.matrix.getRowDimension();
		int otherColumns = other.matrix.getColumnDimension();
		
		if (rows != otherRows)
			throw new DimensionMismatchException(otherRows, rows);
		
		Matrix result = new Matrix(rows, columns + otherColumns);
		
		for (int r = 0; r < rows; r++)
			for (int c = 0; c < columns; c++)
				result.matrix.setEntry(r, c, this.matrix.getEntry(r, c));
		
		for (int r = 0; r < otherRows; r++)
			for (int c = 0; c < otherColumns; c++)
				result.matrix.setEntry(r, columns + c, other.matrix.getEntry(r, c));
		
		return result;
	}
	
	public void setRowWithValue(int row, double value) {
		
		int columns = matrix.getColumnDimension();
		
		for (int c = 0; c < columns; c++)
			this.matrix.setEntry(row, c, value);
	}
	
	public void setColumnWithValue(int column, double value) {
		
		int rows = matrix.getRowDimension();
		
		for (int r = 0; r < rows; r++)
			this.matrix.setEntry(r, column, value);
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((matrix == null) ? 0 : matrix.hashCode());
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
		if (matrix == null) {
			if (other.matrix != null)
				return false;
		} else if (!matrix.equals(other.matrix))
			return false;
		return true;
	}

	@Override
	public String toString() {

		StringBuilder builder = new StringBuilder();
		int rows = matrix.getRowDimension();
		int columns = matrix.getColumnDimension();

		builder.append("[" + rows + ";" + columns + "]\n");
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < columns; c++) {
				builder.append(matrix.getEntry(r, c));
				builder.append('\t');
			}
			builder.append('\n');
		}
		return builder.toString();
	}

	/**
	 * 
	 * @param rows
	 * @param columns
	 * @param min
	 * @param max
	 * @return
	 */
	public static Matrix random(int rows, int columns, double min, double max) {

		Random rng = new Random();
		Matrix m = new Matrix(rows, columns);

		for (int r = 0; r < rows; r++)
			for (int c = 0; c < columns; c++)
				m.matrix.setEntry(r, c, min + (max - min) * rng.nextDouble());

		return m;
	}
}
