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

public class ParamOptimization {

	public static void main(String[] args) {

		try {
			if(args.length == 2) {
				DataSource source = new DataSource(args[1]);
				Instances data = source.getDataSet();
				data.setClassIndex(data.numAttributes()-1);
				
				System.out.println("Barrido de parámetros...");
				barridoParametros(data, args[2],Integer.parseInt(args[0]));			
			
			}else if (args.length == 0) {
				System.out.println("Programa que recoge el arff y obtiene los mejores parámetros para el clasificador y crea el modelo.");
	    		System.out.println("@pre El archivo de arff no esté vacío y la clase es el último del atributo.");
	    		System.out.println("@post Se obtiene el clasificador más óptimo y el archivo .arff no se modifica.");
	    		System.out.println("@param Indicar el porcentaje para la evaluacion Hold-Out.");
	    		System.out.println("@param Ruta del fichero ARFF.");
	    		System.out.println("@param Ruta donde guardar el modelo.");
			}	
			
		}catch (Exception e) {
			e.printStackTrace();
		}

	}
		
	
	private static BayesNet barridoParametros(Instances data,String rutaModelo,int porcentaje) throws Exception{
			
			int minoritaria = indiceClaseMenor(data);
			double fmeasureAct = 0;
			double fmeasureOpt = 0;
	
			RemovePercentage rp = new RemovePercentage();
			rp.setPercentage(66);
	
			rp.setInvertSelection(true);
			rp.setInputFormat(data);
			Instances train = Filter.useFilter(data, rp);
			
			rp.setInvertSelection(false);
			rp.setInputFormat(data);
			Instances test = Filter.useFilter(data, rp);
	
			for(int i=0;i< train.numInstances();i+= (int) Math.round(0.01*train.numInstances())) {
				BayesNet bayesNet = new BayesNet();
	
				bayesNet.setSearchAlgorithm(new SearchAlgorithm());
				bayesNet.setEstimator(new BayesNetEstimator());
				
				bayesNet.buildClassifier(train);
				Evaluation ev = new Evaluation(train);
				
				ev.evaluateModel(bayesNet, test);
				fmeasureAct = ev.fMeasure(minoritaria);
				
				if(fmeasureAct > fmeasureOpt) {
					fmeasureOpt = fmeasureAct;
				}
			}
			
			BayesNet bayesNet = new BayesNet();
			SerializationHelper.write(rutaModelo, bayesNet);
			
			return bayesNet;
		}
	
	private static int indiceClaseMenor(Instances data)throws Exception{
		AttributeStats as = data.attributeStats(data.classIndex());
		Attribute atrb = data.attribute(data.classIndex());
		
		if(atrb.isNominal()) {
			return Utils.minIndex(as.nominalCounts);
		}else if(atrb.isNumeric()) {
			return (int) as.numericStats.min;
		}
		return 0;
	}

}
