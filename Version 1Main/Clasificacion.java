package pack;

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

public class Clasificacion 
{
	
	public Clasificacion(){}
	
	public void BayesMultinomial(String args[]) throws Exception{
	//public void BayesMultinomial(String pRutaTrain, String pRutaTest, String pRutaModelo, String pRutaCalidad) throws Exception{
	//public void BayesMultinomial(Instances pTrain, Instances pTest, String pRutaModelo, String pRutaCalidad) throws Exception{
		Instances train = Clasificacion.cargarInstancias(args[0]);
		Instances test = Clasificacion.cargarInstancias(args[1]);
		Clasificacion.obtenerModeloNBM(train, args[2]);
		//Las dos primeras son usando solo data y las demas con train y test
		
		//Clasificacion.calidadEstimadaNBM_HO(train, 70, pRutaCalidad);
		//Clasificacion.calidadEstimadaNBM_KFCV(train, 70, pRutaCalidad);
		Clasificacion.calidadEstimadaNBM_HO2(train, test, 70, args[3]);
		//Clasificacion.calidadEstimadaNBM_KFCV2(train, test, 70, pRutaCalidad);
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
	
	private static void calidadEstimadaNBM_KFCV(Instances pData, int pFolds, String pRutaCalidad) throws Exception
	{
		NaiveBayesMultinomial clasificador = new NaiveBayesMultinomial(); 
		Evaluation evaluador = new Evaluation(pData);
		evaluador.crossValidateModel(clasificador, pData, pFolds, new Random(1));
		obtenerResultados(evaluador, pRutaCalidad,  "INFORME DE LA CALIDAD ESTIMADA DEL MODELO: CLASIFICADOR = NAIVE BAYES MULTINOMIAL - ESQUEMA DE EVALUACION = K-FOLD CROSS VALIDATION");	
	}
	
	private static void calidadEstimadaNBM_HO(Instances pData, double pPercentage, String pRutaCalidad) throws Exception
	{
		// DIVISION DEL CONJUNTO DE DATOS EN TRAIN Y TEST
		RemovePercentage remove = new RemovePercentage();
		remove.setPercentage(pPercentage);
										
		remove.setInvertSelection(true);
		remove.setInputFormat(pData);
		Instances train = Filter.useFilter(pData, remove);
										
		remove.setInvertSelection(false);
		remove.setInputFormat(pData);
		Instances test = Filter.useFilter(pData, remove);
		
		NaiveBayesMultinomial clasificador = new NaiveBayesMultinomial();
		clasificador.buildClassifier(train);
		Evaluation evaluador = new Evaluation(train);
		evaluador.evaluateModel(clasificador, test);
		obtenerResultados(evaluador, pRutaCalidad, "INFORME DE LA CALIDAD ESTIMADA DEL MODELO: CLASIFICADOR = NAIVE BAYES MULTINOMIAL - ESQUEMA DE EVALUACION = HOLD-OUT");
	}
	
	private static void calidadEstimadaNBM_KFCV2(Instances pTrain, Instances pTest, int pFolds, String pRutaCalidad) throws Exception
	{
		// DATOS YA DIVIDIDOS EN TRAIN Y TEST
		
		pTrain.addAll(pTest);
		NaiveBayesMultinomial clasificador = new NaiveBayesMultinomial(); 
		Evaluation evaluador = new Evaluation(pTrain);
		evaluador.crossValidateModel(clasificador, pTrain, pFolds, new Random(1));
		obtenerResultados(evaluador, pRutaCalidad,  "INFORME DE LA CALIDAD ESTIMADA DEL MODELO: CLASIFICADOR = NAIVE BAYES MULTINOMIAL - ESQUEMA DE EVALUACION = K-FOLD CROSS VALIDATION");	
	}
	
	
	private static void calidadEstimadaNBM_HO2(Instances pTrain, Instances pTest, double pPercentage, String pRutaCalidad) throws Exception
	{
		// DATOS YA DIVIDIDOS EN TRAIN Y TEST
	
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
