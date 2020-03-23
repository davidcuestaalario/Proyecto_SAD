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

		Instances train = getBaselineModel.cargarInstancias(args[0]);

		getBaselineModel.obtenerModeloNBM(train, args[2]);

		getBaselineModel.calidadEstimadaNBM_HO(train, 70, args[2]);
		//getBaselineModel.calidadEstimadaNBM_KFCV(train, 70, args[2]);

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
		Instances dev = Filter.useFilter(pData, remove);
		
		NaiveBayesMultinomial clasificador = new NaiveBayesMultinomial();
		clasificador.buildClassifier(train);
		Evaluation evaluador = new Evaluation(train);
		evaluador.evaluateModel(clasificador, dev);
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