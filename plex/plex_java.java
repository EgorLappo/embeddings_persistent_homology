import java.util.*;
import java.lang.*;
import java.io.*;

import edu.stanford.math.plex4.api.Plex4;
import edu.stanford.math.plex4.examples.PointCloudExamples;
import edu.stanford.math.plex4.homology.barcodes.BarcodeCollection;
import edu.stanford.math.plex4.homology.chain_basis.Simplex;
import edu.stanford.math.plex4.homology.interfaces.AbstractPersistenceAlgorithm;
import edu.stanford.math.plex4.metric.impl.EuclideanMetricSpace;
import edu.stanford.math.plex4.metric.landmark.LandmarkSelector;
import edu.stanford.math.plex4.metric.landmark.MaxMinLandmarkSelector;
import edu.stanford.math.plex4.streams.impl.WitnessStream;

public class plex_java {

	public static void main(String[] args) throws java.io.FileNotFoundException {

		int n = 20;
		int d = 4;
		int numLandmarkPoints = 70;
		double maxDistance = 0.1;

		Scanner sc = new Scanner(new BufferedReader(new FileReader("plex_input/test_word2vec_plex")));
		int rows = 0;
		int columns = 0;
		while(sc.hasNextLine()) {
			++rows;
			String[] line = sc.nextLine().trim().split(" ");
			columns = line.length;
		}
		sc.close();
		
		double[][] points = new double[rows][columns];

		//System.out.println("Preread the point cloud!");

		// read in the data
		Scanner sc2 = new Scanner(new BufferedReader(new FileReader("plex_input/test_word2vec_plex")));
		while(sc2.hasNextLine()) {
         	for (int i=0; i<points.length; i++) {
	            String[] line = sc2.nextLine().trim().split(" ");
	            for (int j=0; j<line.length; j++) {
	               points[i][j] = Double.parseDouble(line[j]);
	            }
        	}
      	}
		sc2.close();

		//System.out.println("Read in the point cloud!");

		EuclideanMetricSpace metricSpace = new EuclideanMetricSpace(points);
		
		LandmarkSelector<double[]> landmarkSelector = new MaxMinLandmarkSelector<double[]>(metricSpace, numLandmarkPoints);
		
		double R = landmarkSelector.getMaxDistanceFromPointsToLandmarks();

		maxDistance = R / 8;

		WitnessStream<double[]> stream = new WitnessStream<double[]>(metricSpace, landmarkSelector, d + 1, maxDistance, n);
		stream.finalizeStream();
		
		System.out.println("Number of simplices in complex: " + stream.getSize());
		
		AbstractPersistenceAlgorithm<Simplex> algorithm = Plex4.getDefaultSimplicialAlgorithm(d + 1);
		
		BarcodeCollection<Double> intervals = algorithm.computeIntervals(stream);
		
		System.out.println(intervals);
	}
}
