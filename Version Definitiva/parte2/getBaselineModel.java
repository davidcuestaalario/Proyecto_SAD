package parte2;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Random;
import java.util.Vector;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;
import weka.core.converters.ConverterUtils.DataSource;

public class getBaselineModel 
{
	
	public getBaselineModel(){}
	
	public static void main(String args[]) throws Exception{
		
		if(args.length == 5) {

			Instances train = getBaselineModel.cargarInstancias(args[0]);
			Instances dev = getBaselineModel.cargarInstancias(args[1]);
	
			getBaselineModel.obtenerModeloNBM(train, args[2]);
	
			getBaselineModel.calidadEstimadaNBM_KFCV(train, dev, 70, args[3]);
			getBaselineModel.calidadEstimadaNBM_HO(train, dev, 70, args[4]);
			
		}else if (args.length == 0) {
			System.out.println("Programa que obtiene la calidad estimada a partir del ARFF mediante Naive Bayes Multinomial");
    		System.out.println("@pre La clase es el ultimo atributo");
    		System.out.println("@post Se han generado los ficheros del modelo y la calidad");
    		System.out.println("@param Ruta del fichero train ARFF");
    		System.out.println("@param Ruta del fichero test ARFF");
    		System.out.println("@param Ruta del modelo");
    		System.out.println("@param Ruta donde guardar la calidad estimada de la evaluacion K-Fold Cross-Validation.");
    		System.out.println("@param Ruta donde guardar la calidad estimada de la evaluacion No Honesta.");
		}

	}
	
	public void BayesNetwork(){}
	
	private static Instances cargarInstancias(String pRutaARFF) throws Exception
	{
		DataSource source = new DataSource(pRutaARFF);
		Instances data = source.getDataSet();		
		data.randomize(new Random());			
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}
	
	private static void obtenerModeloNBM(Instances pData, String pRutaModelo) throws Exception
	{
		NaiveBayesMultinomial clasificador = new NaiveBayesMultinomial(); 
		clasificador.buildClassifier(pData);
		Vector<Object> v = new Vector<>();
		v.add(clasificador);
		v.add(new Instances(pData, 0));
		SerializationHelper.write(pRutaModelo, v);	
	}
	
	private static void calidadEstimadaNBM_KFCV(Instances pTrain, Instances pTest, int pFolds, String pRutaCalidad) throws Exception
	{
		
		pTrain.addAll(pTest);
		NaiveBayesMultinomial clasificador = new NaiveBayesMultinomial(); 
		Evaluation evaluador = new Evaluation(pTrain);
		evaluador.crossValidateModel(clasificador, pTrain, pFolds, new Random(1));
		obtenerResultados(evaluador, pRutaCalidad,  "INFORME DE LA CALIDAD ESTIMADA DEL MODELO: CLASIFICADOR = NAIVE BAYES MULTINOMIAL - ESQUEMA DE EVALUACION = K-FOLD CROSS VALIDATION");	
	}
	
	
	private static void calidadEstimadaNBM_HO(Instances pTrain, Instances pTest, double pPercentage, String pRutaCalidad) throws Exception
	{
	
		NaiveBayesMultinomial clasificador = new NaiveBayesMultinomial();
		clasificador.buildClassifier(pTrain);
		Evaluation evaluador = new Evaluation(pTrain);
		evaluador.evaluateModel(clasificador, pTest);
		obtenerResultados(evaluador, pRutaCalidad, "INFORME DE LA CALIDAD ESTIMADA DEL MODELO: CLASIFICADOR = NAIVE BAYES MULTINOMIAL - ESQUEMA DE EVALUACION = HOLD-OUT");
	}
	
	
	private static void obtenerResultados(Evaluation pEvaluador, String pRutaResultados, String pCabecera)
	{
		try 
		{
			BufferedWriter bw = new BufferedWriter(new FileWriter(pRutaResultados));
			bw.write(pCabecera);
			bw.newLine();
			bw.write(pEvaluador.toSummaryString());
			bw.newLine();
			bw.write(pEvaluador.toClassDetailsString());
			bw.newLine();
			bw.write(pEvaluador.toMatrixString());
			bw.close();
		} 
		catch (Exception e) 
		{
			System.out.println("ERROR AL ESCRIBIR LOS RESULTADOS");
		}
	}
	
}