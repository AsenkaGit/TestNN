package fr.asenka.detektor.util;


import static fr.asenka.detektor.util.Matrix.binaryMatrix;
import static fr.asenka.detektor.util.Matrix.sum;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class MatrixTest {
	
	private final double DELTA = 0.000000001d;

	private int index;

	@Test
	void testNewMatrix() {
		
		Matrix m = new Matrix("1 2 3 ; 4 5 6 ; 7 8 9");
		
		assertEquals(3, m.rows());
		assertEquals(3, m.columns());
		assertEquals(1d, m.get(0, 0), DELTA);
		assertEquals(2d, m.get(0, 1), DELTA);
		assertEquals(3d, m.get(0, 2), DELTA);
		assertEquals(4d, m.get(1, 0), DELTA);
		assertEquals(5d, m.get(1, 1), DELTA);
		assertEquals(6d, m.get(1, 2), DELTA);
		assertEquals(7d, m.get(2, 0), DELTA);
		assertEquals(8d, m.get(2, 1), DELTA);
		assertEquals(9d, m.get(2, 2), DELTA);
		
		Matrix p = new Matrix(new double[][] {{1d, 2d, 3d}, {4d, 5d, 6d}, {7d, 8d, 9d}});
		
		assertEquals(p, m);
	}
	
	@Test
	void testCopy() {
		Matrix m = new Matrix("1 2 3 ; 4 5 6 ; 7 8 9");
		Matrix c = m.copy();
		
		assertEquals(m, c);
		assertNotSame(m, c);
		assertNotSame(m.getRawData(), c.getRawData());
	}
	
	@Test
	void testTranspose() {
		
		Matrix m = new Matrix("1 2 3 ; 4 5 6");
		Matrix expected = new Matrix("1 4 ; 2 5 ; 3 6");
		
		assertEquals(expected, m.transpose());
	}
	
	@Test
	void testNegative() {
		
		Matrix m = new Matrix("1 2 3 ; 4 5 6");
		Matrix expected = new Matrix("-1 -2 -3 ; -4 -5 -6");
		
		assertEquals(expected, m.negative());
		assertEquals(m.negative(), m.multiply(-1d));
	}
	
	@Test
	void testFlat() {
		
		Matrix m = new Matrix("1 2 3 ; 4 5 6");
		
//		assertEquals(new Matrix("1 2 3 4 5 6"), m.flatHorizontaly());
		assertEquals(new Matrix("1;2;3;4;5;6"), m.flatColumn());
	}
	
	@Test
	void testAdd() {
		Matrix m1 = new Matrix("1 1 1 ; 1 1 1");
		Matrix m2 = new Matrix("0 1 0 ; 2 2 2");
		Matrix expected = new Matrix("1 2 1 ; 3 3 3");
		
		assertEquals(expected, m1.add(m2));
		assertEquals(expected, m2.add(m1));
	}
	
	@Test
	void testSubtract() {
		Matrix m1 = new Matrix("1 1 1 ; 1 1 1");
		Matrix m2 = new Matrix("0 1 0 ; 2 2 2");
		Matrix expectedM1sM2 = new Matrix("1 0 1 ; -1 -1 -1");
		Matrix expectedM2sM1 = new Matrix("-1 0 -1 ; 1 1 1");
		
		assertEquals(expectedM1sM2, m1.subtract(m2));
		assertEquals(expectedM2sM1, m2.subtract(m1));
	}
	
	@Test
	void testMultiply() {
		
		Matrix m1 = new Matrix("1 2 3 ; 4 5 6");
		Matrix m2 = new Matrix("2 4 ; 3 0 ; 1 0");
		Matrix expectedM1xM2 = new Matrix("11 4 ; 29 16");
		Matrix expectedM2xM1 = new Matrix("18 24 30 ; 3 6 9 ; 1 2 3");
		
		assertEquals(expectedM1xM2, m1.multiply(m2));
		assertEquals(expectedM2xM1, m2.multiply(m1));
	}
	
	@Test
	void testScalarAdd() {
		
		Matrix m = new Matrix("0 1 0 ; 2 2 2");
		Matrix expected = new Matrix("10 11 10 ; 12 12 12");
		
		assertEquals(expected, m.add(10d));
	}

	@Test
	void testScalarSubtract() {
		
		Matrix m = new Matrix("10 11 10 ; 12 12 12");
		Matrix expected = new Matrix("0 1 0 ; 2 2 2");
		
		assertEquals(expected, m.subtract(10d));
	}

	@Test
	void testScalarMultiply() {
		
		Matrix m = new Matrix("0 1 0 ; 2 2 2");
		Matrix expected = new Matrix("0 10 0 ; 20 20 20");
		
		assertEquals(expected, m.multiply(10d));
	}
	
	@Test
	void testMultiplyEachEntry() {
		
		Matrix m1 = new Matrix("1 1 1 ; 2 2 2");
		Matrix m2 = new Matrix("0 1 3 ; 0 1 3");
		Matrix expected = new Matrix("0 1 3 ; 0 2 6");
		
		assertEquals(expected, m1.multiplyEachEntry(m2));
		assertEquals(expected, m2.multiplyEachEntry(m1));
	}
	
	@Test
	void testGetSubMatrix() {
		
		Matrix m = new Matrix("1 2 3 4 5 ; 1 2 3 4 5 ; 1 2 3 4 5 ; 1 2 3 4 5");
		Matrix expected = new Matrix("2 3 4 5 ; 2 3 4 5 ; 2 3 4 5");

		assertEquals(expected, m.getSubMatrix(1, 1));
	}
	
	@Test
	void testMin() {
		
		Matrix m = new Matrix("1 2 3 4 5 ; 0 0 0 1 0 ; 10 11 12 13 14");
		
		assertEquals(0, m.min());
		assertEquals(new Matrix("0 0 0 1 0"), m.minByColumn());
		assertEquals(new Matrix("1 ; 0 ; 10"), m.minByRow());
	}
	
	@Test
	void testIndexMin() {
		
		Matrix m = new Matrix("23 1 8 ; 0 1 0 ; 1 9 51");
		
		assertEquals(new Matrix("1 0 1"), m.indexMinByColumn());
		assertEquals(new Matrix("1;0;0"), m.indexMinByRow());
	}
	
	@Test
	void testMax() {
		
		Matrix m = new Matrix("1 2 3 4 5 ; 0 0 0 1 0 ; 10 11 12 13 14");
		
		assertEquals(14, m.max());
		assertEquals(new Matrix("10 11 12 13 14"), m.maxByColumn());
		assertEquals(new Matrix("5 ; 1 ; 14"), m.maxByRow());	
	}
	
	@Test
	void testIndexMax() {
		
		Matrix m = new Matrix("23 1 8 ; 0 1 0 ; 1 9 51");
		
		assertEquals(new Matrix("0 2 2"), m.indexMaxByColumn());
		assertEquals(new Matrix("0;1;2"), m.indexMaxByRow());
	}
	
	@Test
	void testApplyOnEach() {
		
		Matrix m = new Matrix("1 1 ; 2 2");
		Matrix expected = new Matrix("2.25 2.25 ; 4.25 4.25");
		
		assertEquals(expected, m.applyOnEach(d -> d * 2 + 0.25));
	}
	
	@Test
	void testBinaryMatrix() {
		
		Matrix rowMatrix = new Matrix("0 1 4 2 3 1");
		Matrix expected = new  Matrix(
				  "1 0 0 0 0 0;"
				+ "0 1 0 0 0 1;"
				+ "0 0 0 1 0 0;"
				+ "0 0 0 0 1 0;"
				+ "0 0 1 0 0 0"
		);
		assertEquals(expected, binaryMatrix(rowMatrix, 5));
	}
	
	@Test
	void testStream() {
		
		Matrix m = new Matrix("1 2 3 4 5 ; 6 7 8 9 10");
		
		assertEquals(m.size(), m.stream().count());
		assertEquals(1, m.stream().min((a, b) -> Double.compare(a, b)).get());
		assertEquals(10, m.stream().max((a, b) -> Double.compare(a, b)).get());
		assertEquals(m.min(), m.stream().min((a, b) -> Double.compare(a, b)).get());
		assertEquals(m.max(), m.stream().max((a, b) -> Double.compare(a, b)).get());
	}
	
	@Test
	void testForEachColumn() {
		Matrix m = new Matrix("1 2 3 4 5 ; 0 0 0 1 0 ; 10 11 12 13 14");
		index = 0;
		m.forEachColumn(c -> assertEquals(m.getColumn(index++), c));
	}
	
	@Test
	void testForEachRow() {
		Matrix m = new Matrix("1 2 3 4 5 ; 0 0 0 1 0 ; 10 11 12 13 14");
		index = 0;
		m.forEachRow(r -> assertEquals(m.getRow(index++), r));
	}
	
	@Test
	void testSum() {
		Matrix m = new Matrix("1 2 3 4 5 ; 0 0 0 1 0 ; 10 11 12 13 14");
		
		assertEquals(new Matrix("11 13 15 18 19"), sum(m));
		assertTrue(sum(m).isRow());
		assertEquals(new Matrix("76"), sum(sum(m)));
		assertTrue(sum(sum(m)).isScalar());
	}

	@Test
	void testIterator() {
		
		Matrix m = new Matrix("1 2 3 4 5 ; 6 7 8 9 10");
		double num = 1d;
		
		for (Double e : m) 
			assertEquals(num++, e);
	}
	
	@Test
	void testRandom() {
		
		Matrix random = Matrix.random(5, 4, 0d, 1d);
		
		assertEquals(5, random.rows());
		assertEquals(4, random.columns());
		random.forEach(e -> assertEquals(0.5d, e, 0.5d));
	}
	
	 @Test
	 void testEquals() {
		 
		 Matrix m1 = new Matrix("1 2 3 4 5 ; 6.00 7 8 9 10");
		 Matrix m2 = new Matrix("1 2 3 4 5 ; 6.01 7 8 9 10");
		 
		 assertNotEquals(m1, m2);
		 assertEquals(m1, m1.copy());
		 assertEquals(m2, m2.copy());
	 }
}
