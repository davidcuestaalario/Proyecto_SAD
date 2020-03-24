import java.util.ArrayList;
import java.util.Vector;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.estimate.SimpleEstimator;
import weka.classifiers.bayes.net.search.SearchAlgorithm;
import weka.classifiers.bayes.net.search.local.HillClimber;
import weka.classifiers.bayes.net.search.local.K2;
import weka.classifiers.bayes.net.search.local.LAGDHillClimber;
import weka.classifiers.bayes.net.search.local.RepeatedHillClimber;
import weka.classifiers.bayes.net.search.local.SimulatedAnnealing;
import weka.classifiers.bayes.net.search.local.TAN;
import weka.classifiers.bayes.net.search.local.TabuSearch;
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
				DataSource source = new DataSource(args[0]);
				Instances data = source.getDataSet();
				data.setClassIndex(data.numAttributes()-1);
				
				System.out.println("Barrido de parámetros...");
				Vector<Object> v = barridoParametros(data);			
			
				SearchAlgorithm algoritmo = (SearchAlgorithm) v.get(0);
				Boolean tree = (Boolean) v.get(1);
				
				guardarModelo(args[1],algoritmo,tree.booleanValue());
				System.out.println("Barrido de parámetros terminado");
			}else if (args.length == 0) {
				System.out.println("Programa que recoge el ARFF y obtiene los mejores parámetros para el clasificador y crea el modelo");
	    		System.out.println("@pre El archivo de arff no esté vacío y la clase es el último del atributo");
	    		System.out.println("@post Se obtiene el clasificador más óptimo y el archivo .ARFF no se modifica");
	    		System.out.println("@param Ruta del fichero train ARFF");
	    		System.out.println("@param Ruta donde guardar el modelo");
	    		
			}	
			
		}catch (Exception e) {
			e.printStackTrace();
		}

	}
		
	
	private static Vector<Object> barridoParametros(Instances data) throws Exception{
			
			int minoritaria = indiceClaseMenor(data);
			double fmeasureAct = 0;
			double fmeasureOpt = 0;
	
			RemovePercentage rp = new RemovePercentage();
			rp.setPercentage(70);
	
			rp.setInvertSelection(true);
			rp.setInputFormat(data);
			Instances train = Filter.useFilter(data, rp);
			
			rp.setInvertSelection(false);
			rp.setInputFormat(data);
			Instances test = Filter.useFilter(data, rp);
			
			ArrayList<Object> algoritmosBusqueda = new ArrayList<Object>();
			algoritmosBusqueda.add(new K2());
			algoritmosBusqueda.add(new TAN());
			algoritmosBusqueda.add(new HillClimber());
			algoritmosBusqueda.add(new LAGDHillClimber());
			algoritmosBusqueda.add(new RepeatedHillClimber());
			algoritmosBusqueda.add(new SimulatedAnnealing());
			algoritmosBusqueda.add(new TabuSearch());

			SearchAlgorithm algoritmoOptimo=null;
			boolean treeOptimo=false;
			
			BayesNet bayesNet = new BayesNet();
				
			bayesNet.setEstimator(new SimpleEstimator());
			bayesNet.buildClassifier(train);
			Evaluation ev = new Evaluation(train);

			for(int j=0; j <algoritmosBusqueda.size();j++) {
					
				SearchAlgorithm algoritmo = (SearchAlgorithm) algoritmosBusqueda.get(j);
				bayesNet.setSearchAlgorithm(algoritmo);
				for(int k =0; k<2;k++) {
					if(k==0) {
						bayesNet.setUseADTree(true);
					}else {
						bayesNet.setUseADTree(false);
					}
					ev.evaluateModel(bayesNet, test);
					fmeasureAct = ev.fMeasure(minoritaria);
				
					if(fmeasureAct > fmeasureOpt) {
						fmeasureOpt = fmeasureAct;
						if(k==0) {
							treeOptimo = true;
						}else {
							treeOptimo = false;
						}
						algoritmoOptimo = algoritmo;
						}
					}
				}
			
			Vector<Object> v = new Vector<Object>();
			Boolean treeOptimoObject = Boolean.valueOf(treeOptimo);
			v.add(0,algoritmoOptimo);
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
	
	private static void guardarModelo(String rutaModelo, SearchAlgorithm algorithm ,boolean tree) throws Exception{
		BayesNet bayesNet = new BayesNet();
		
		bayesNet.setEstimator(new SimpleEstimator());
		bayesNet.setUseADTree(tree);
		bayesNet.setSearchAlgorithm(algorithm);
		
		SerializationHelper.write(rutaModelo, bayesNet);
	}

}
