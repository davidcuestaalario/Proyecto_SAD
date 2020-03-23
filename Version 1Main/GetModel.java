import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.estimate.BayesNetEstimator;
import weka.classifiers.bayes.net.search.SearchAlgorithm;
import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

public class GetModel {

	public static void main(String[] args) {

		try {
			
			if(args.length == 3) {
				DataSource source = new DataSource(args[0]);
				Instances data = source.getDataSet();
				data.setClassIndex(data.numAttributes()-1);
								
				DataSource source2 = new DataSource(args[1]);
				Instances dataTest = source2.getDataSet();
				data.setClassIndex(data.numAttributes()-1);
				
				System.out.println("Realizando cross validation...");
				crossValidation(args[2], dataTest, args[3]);
				System.out.println("Realizando no honesta...");
				noHonesta(args[2], dataTest, args[4]);

			}else if (args.length == 0) {
				System.out.println("Programa que obtiene la calidad estimada a partir del arff.");
	    		System.out.println("@pre La clase es el ultimo atributo.");
	    		System.out.println("@post Se han generado los ficheros del modelo y la calidad.");
	    		System.out.println("@param Ruta del fichero train ARFF.");
	    		System.out.println("@param Ruta del fichero test ARFF.");
	    		System.out.println("@param Ruta del modelo.");
	    		System.out.println("@param Ruta donde guardar la calidad estimada para la evaluacion K-Fold Cross Validation.");
	    		System.out.println("@param Ruta donde guardar la calidad estimada para la evaluacion No-Honesta.");
			}
			
		}catch (Exception e) {
			e.printStackTrace();
		}

	}
	
	private static void crossValidation(String rutaModelo,Instances test,String rutaResultado) throws Exception{
		
		BayesNet bayesNet = (BayesNet) SerializationHelper.read(rutaModelo);
		
		bayesNet.buildClassifier(test);
		bayesNet.setEstimator(new BayesNetEstimator());
		bayesNet.setSearchAlgorithm(new SearchAlgorithm());
		
		Evaluation ev = new Evaluation(test);
		ev.crossValidateModel(bayesNet, test, 10, new Random(1));
		
		getResultados(rutaResultado,ev,false);
	}
	
	private static void noHonesta(String rutaModelo,Instances test,String rutaResultado) throws Exception{
		
		BayesNet bayesNet = (BayesNet) SerializationHelper.read(rutaModelo);
		
		bayesNet.buildClassifier(test);
		bayesNet.setEstimator(new BayesNetEstimator());
		bayesNet.setSearchAlgorithm(new SearchAlgorithm());
		
		Evaluation ev = new Evaluation(test);
		ev.evaluateModel(bayesNet, test);
		
		getResultados(rutaResultado,ev,true);
	}

	private static void getResultados(String rutaResultado,Evaluation ev,boolean evaluar)throws Exception {
		
		BufferedWriter bw = new BufferedWriter(new FileWriter(rutaResultado));
		
		if(!evaluar) {
			bw.append("RESULTADOS DEL CROSS-VALIDATION");
		}else {
			bw.append("RESULTADOS DEL NO HONESTA");
		}
		
		bw.append("\n Correct: " + ev.pctCorrect());
		bw.append("\n Incorrect: " + ev.pctIncorrect());
		
		bw.append("\n F-Measure: " + ev.weightedFMeasure());
		bw.append("\n Precision: " + ev.weightedPrecision());
		bw.append("\n Recall: "+ ev.weightedRecall());
		bw.close();
		
	}

}
