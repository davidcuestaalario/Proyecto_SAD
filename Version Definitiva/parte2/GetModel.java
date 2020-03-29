import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.estimate.BayesNetEstimator;
import weka.classifiers.bayes.net.estimate.SimpleEstimator;
import weka.classifiers.bayes.net.search.SearchAlgorithm;
import weka.classifiers.bayes.net.search.local.K2;
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
			
			if(args.length == 4) {
				
				String ficheroTrain = args[0];
				
				boolean filtrado = false;
				if(ficheroTrain.contains("Filtrados")) {
					filtrado=true;
				}
				
				System.out.println("Cargando datos...");
				DataSource source = new DataSource(ficheroTrain);
				Instances data = source.getDataSet();
				
				if(filtrado) {
					data.setClassIndex(data.numAttributes()-1);
				}else {
					data.setClassIndex(0);
				}
				System.out.println("Datos cargados...");
			
				System.out.println("Realizando cross-validation...");
				crossValidation(args[1], data, args[2]);
				System.out.println("Cross-Validation terminado");
				
				System.out.println("Realizando no honesta...");
				noHonesta(args[1], data, args[3]);
				System.out.println("No honesta terminado");

			}else if (args.length == 0) {
				System.out.println("Programa que obtiene la calidad estimada a partir del train ARFF");
	    		System.out.println("@pre La clase es el ultimo atributo");
	    		System.out.println("@post Se han generado los ficheros del modelo y la calidad");
	    		System.out.println("@param Ruta del fichero train ARFF");
	    		System.out.println("@param Ruta del modelo donde guardar ");
	    		System.out.println("@param Ruta donde guardar la calidad estimada de la evaluacion K-Fold Cross-Validation.");
	    		System.out.println("@param Ruta donde guardar la calidad estimada de la evaluacion No Honesta.");
	    		
			}
			
		}catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	private static void crossValidation(String rutaModelo,Instances train,String rutaResultado) throws Exception{	
		BayesNet bayesNet = (BayesNet) SerializationHelper.read(rutaModelo);
		bayesNet.buildClassifier(train);
		
		Evaluation ev = new Evaluation(train);
		ev.crossValidateModel(bayesNet, train, 10, new Random(1));
		
		getResultados(rutaResultado,ev,false);
	}
	
	private static void noHonesta(String rutaModelo,Instances train,String rutaResultado) throws Exception{
		
		BayesNet bayesNet = (BayesNet) SerializationHelper.read(rutaModelo);		
		bayesNet.buildClassifier(train);
		
		Evaluation ev = new Evaluation(train);
		ev.evaluateModel(bayesNet, train);
		
		getResultados(rutaResultado,ev,true);
	}

	private static void getResultados(String rutaResultado,Evaluation ev,boolean evaluar)throws Exception {
		
		BufferedWriter bw = new BufferedWriter(new FileWriter(rutaResultado));
		
		if(!evaluar) {
			bw.append("RESULTADOS DEL CROSS-VALIDATION\n");
		}else {
			bw.append("RESULTADOS DEL NO HONESTA\n");
		}
		bw.append(ev.toSummaryString());
		bw.newLine();
		bw.append(ev.toClassDetailsString());
		bw.newLine();
		bw.append(ev.toMatrixString());
		bw.close();	
	}

}
