import java.util.Vector;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.estimate.SimpleEstimator;
import weka.classifiers.bayes.net.search.local.K2;
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
			if(args.length == 3) {
				
				String ficheroTrain = args[0];
				String fichData70 = args[1];
				String fichData30 = args[2];
				
				System.out.println("Cargando datos...");
				
				DataSource source = new DataSource(ficheroTrain);
				Instances data = source.getDataSet();
				if(esFicheroFiltrado(ficheroTrain)) {
					data.setClassIndex(data.numAttributes()-1);
				}else {
					data.setClassIndex(0);
				}
				
				DataSource source70 = new DataSource(fichData70);
				Instances data70 = source70.getDataSet();
				if(esFicheroFiltrado(ficheroTrain)) {
					data70.setClassIndex(data.numAttributes()-1);
				}else {
					data70.setClassIndex(0);
				}
				
				DataSource source30 = new DataSource(fichData30);
				Instances data30 = source30.getDataSet();
				if(esFicheroFiltrado(ficheroTrain)) {
					data30.setClassIndex(data.numAttributes()-1);
				}else {
					data30.setClassIndex(0);
				}
				System.out.println("Datos cargados");
				
				System.out.println("Realizando barrido de parámetros...");
				Vector<Object> v = barridoParametros(data,data70,data30);			
			
				Integer maxPadresInteger = (Integer) v.get(0);
				Boolean tree = (Boolean) v.get(1);
				
				guardarModelo(args[2],maxPadresInteger.intValue(),tree.booleanValue());
				System.out.println("Barrido de parámetros terminado");
				
			}else if (args.length == 0) {
				System.out.println("Programa que recoge el ARFF train, con todas las instancias y dividio entre 70-30, y obtiene los mejores parámetros para el clasificador y crea el modelo más óptimo");
	    		System.out.println("@pre El archivo de arff no esté vacío y la clase es el último del atributo");
	    		System.out.println("@post Se obtiene el clasificador más óptimo y el archivo .ARFF no se modifica");
	    		System.out.println("@param Ruta del fichero train ARFF completo");
	    		System.out.println("@param Ruta del fichero train ARFF 70% de instancias");
	    		System.out.println("@param Ruta del fichero train ARFF 30% de instancias");
	    		System.out.println("@param Ruta donde guardar el modelo");
	    		
			}	
			
		}catch (Exception e) {
			e.printStackTrace();
		}
		
	}
	
	private static boolean esFicheroFiltrado(String fichero) {
		if(fichero.contains("Filtrados")) {
			return true;
		}
		return false;
	}
		
	
	private static Vector<Object> barridoParametros(Instances data,Instances train70, Instances train30) throws Exception{
			
		int minoritaria = indiceClaseMenor(data);
		double fmeasureAct = 0;
		double fmeasureOpt = 0;
		int maxPadres = 0;
		boolean treeOptimo=false;
		
		for(int i=0; i< train70.numAttributes();i++) {
			BayesNet bayesNet = new BayesNet();
						
			
			K2 k2 = new K2();
			k2.setMaxNrOfParents(i);
			SimpleEstimator estimator = new SimpleEstimator();
			estimator.setAlpha(0.6);
			
			bayesNet.setEstimator(estimator);
			bayesNet.setSearchAlgorithm(k2);
			bayesNet.buildClassifier(train70);
			
			for(int k =0; k<2;k++) {
				if(k==0) {
					bayesNet.setUseADTree(true);
				}else {
					bayesNet.setUseADTree(false);
				}
			
				Evaluation ev = new Evaluation(train70);
				ev.evaluateModel(bayesNet, train30);
				fmeasureAct = ev.fMeasure(minoritaria);
				
				if(fmeasureOpt < fmeasureAct) {
					fmeasureOpt = fmeasureAct;
					maxPadres = i;
					if(k==0) {
						treeOptimo = true;
					}else {
						treeOptimo = false;
					}
				}
			}
		}
		
		Vector<Object> v = new Vector<Object>();
		Boolean treeOptimoObject = Boolean.valueOf(treeOptimo);
		v.add(0,new Integer(maxPadres));
		v.add(1,treeOptimoObject);
	
		return v;	
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
	
	private static void guardarModelo(String rutaModelo, int maxPadres ,boolean tree) throws Exception{
		BayesNet bayesNet = new BayesNet();
		
		SimpleEstimator estimator = new SimpleEstimator();
		estimator.setAlpha(0.6);
		bayesNet.setEstimator(estimator);
		
		K2 k2Optimo = new K2();
		k2Optimo.setMaxNrOfParents(maxPadres);
		bayesNet.setUseADTree(tree);
		
		SerializationHelper.write(rutaModelo, bayesNet);
	}

}
