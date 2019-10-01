package fr.asenka.detektor.util;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * 
 * @author 
 *
 */
public final class CustomMatrixUtils {

	/**
	 * 
	 * @param v
	 * @param m
	 * @return
	 */
	public static final double[][] mergeHorizontaly(double[] v, double[][] m) {
		
		if (v == null)
			throw new IllegalArgumentException("null array forbidden");
		
		double[][] m1 =  new double[v.length][1];
		int height = v.length;
		
		for (int row = 0; row < height; row++)
			m1[row][0] = v[row];
		
		return mergeHorizontaly(m1, m);
	}
	
	/**
	 * 
	 * @param m1
	 * @param m2
	 * @return
	 */
	public static final double[][] mergeHorizontaly(double[][] m1, double[][] m2) {
		
		if (m1 == null || m2 == null)
			throw new IllegalArgumentException("null array forbidden");
		
		if (m1.length != m2.length)
			throw new IllegalArgumentException("m1 and m2 must have the same height");
		
		final int height = m1.length;
		final double[][] result = new double[height][];
		
		for (int row = 0; row < height; row++)
			result[row] = ArrayUtils.addAll(m1[row], m2[row]);
		
		return result;
	}
	
	/**
	 * 
	 * @param v
	 * @param m
	 * @return
	 */
	public static final RealMatrix mergeHorizontaly(RealVector v, RealMatrix m) {
		final double[][] mergedValues = mergeHorizontaly(v.toArray(), m.getData());
		return MatrixUtils.createRealMatrix(mergedValues);
	}
	
	/**
	 * 
	 * @param m1
	 * @param m2
	 * @return
	 */
	public static final RealMatrix mergeHorizontaly(RealMatrix m1, RealMatrix m2) {
		final double[][] mergedValues = mergeHorizontaly(m1.getData(), m2.getData());
		return MatrixUtils.createRealMatrix(mergedValues);
	}
	
	/**
	 * 
	 * @param m
	 * @return
	 */
	public static final double[] vectorizeMatrix(RealMatrix m) {
		
		final int rows = m.getRowDimension();
		final int columns = m.getColumnDimension();
		final double[] vectorizedMatrix = new double[rows * columns];
		
		for (int row = 0; row < rows; row++)
			for (int col = 0; col < columns; col++)
				vectorizedMatrix[columns * row + col] = m.getEntry(row, col);
		
		return vectorizedMatrix;
	}
	
	/**
	 * 
	 * @param matrices
	 * @return
	 */
	public static final double[] vectorizeMatrices(RealMatrix... matrices) {
		
		double[] vectorizedMatrices = null;
		
		for (RealMatrix m : matrices) 
			vectorizedMatrices = ArrayUtils.addAll(vectorizedMatrices, vectorizeMatrix(m));
		
		return vectorizedMatrices;
	}
	
	/**
	 * 
	 * @param m1
	 * @param m2
	 * @return
	 */
	public static final RealMatrix scalarMultiply(RealMatrix m1, RealMatrix m2) {
		
		if (!hasSameDimensions(m1, m2))
			throw new IllegalArgumentException("m1 and m2 don't have the same sizes");
		
		int rows = m1.getRowDimension();
		int columns = m1.getColumnDimension();
		
		RealMatrix result = MatrixUtils.createRealMatrix(rows, columns);
		
		for (int row = 0; row < rows; row++)
			for (int col = 0; col < columns; col++)
				result.setEntry(row, col, m1.getEntry(row, col) * m2.getEntry(row, col));
		
		return result;
	}
	
	/**
	 * 
	 * @param m1
	 * @param m2
	 * @return
	 */
	public static final boolean hasSameDimensions(RealMatrix m1, RealMatrix m2) {
		
		int m1Rows = m1.getRowDimension();
		int m2Rows = m2.getRowDimension();
		int m1Columns = m1.getColumnDimension();
		int m2Columns = m2.getColumnDimension();
		
		return m1Rows == m2Rows && m1Columns == m2Columns;
	}
	
	/**
	 * 
	 * @param rows
	 * @param columns
	 * @return
	 */
	public static final RealMatrix createZeroMatrix(int rows, int columns) {
		
		return createMatrixWithValue(rows, columns, 0d);
	}
	
	/**
	 * 
	 * @param rows
	 * @param columns
	 * @param value
	 * @return
	 */
	public static final RealMatrix createMatrixWithValue(int rows, int columns, double value) {
		
		double[][] values = new double[rows][columns];
		
		for (int row = 0; row < rows; row++)
			for (int col = 0; col < columns; col++)
				values[row][col] = value;
		
		return MatrixUtils.createRealMatrix(values);
	}
	
	/**
	 * 
	 * @param m
	 * @param firstRowsRemoved
	 * @param firstColumnsRemoved
	 * @return
	 */
	public static final RealMatrix getSubMatrix(RealMatrix m, int firstRowsRemoved, int firstColumnsRemoved) {
		
		return m.getSubMatrix(firstRowsRemoved, m.getRowDimension() - 1, firstColumnsRemoved, m.getColumnDimension() - 1);
	}
	
	public static final void display(RealMatrix m) {
		display(m.getData());
	}
	
	public static final void display(double[] v) {
		System.out.println(ArrayUtils.toString(v));
	}
	
	public static final void display(double[][] m) {
		int height = m.length;
		for (int row = 0; row < height; row++)
			System.out.println(ArrayUtils.toString(m[row]));
	}
}
