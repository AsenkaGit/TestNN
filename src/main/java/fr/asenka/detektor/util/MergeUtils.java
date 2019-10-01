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
public final class MergeUtils {

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
		
		int height = m1.length;
		
		double[][] result = new double[height][];
		
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
		double[][] mergedValues = mergeHorizontaly(v.toArray(), m.getData());
		return MatrixUtils.createRealMatrix(mergedValues);
	}
	
	/**
	 * 
	 * @param m1
	 * @param m2
	 * @return
	 */
	public static final RealMatrix mergeHorizontaly(RealMatrix m1, RealMatrix m2) {
		double[][] mergedValues = mergeHorizontaly(m1.getData(), m2.getData());
		return MatrixUtils.createRealMatrix(mergedValues);
	}
	
	
//	public static final void display(double[] v) {
//		System.out.println(ArrayUtils.toString(v));
//	}
//	
//	public static final void display(double[][] m) {
//		int height = m.length;
//		for (int row = 0; row < height; row++)
//			System.out.println(ArrayUtils.toString(m[row]));
//	}
}
