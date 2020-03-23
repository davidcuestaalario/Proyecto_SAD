import java.io.BufferedWriter;
import java.io.FileWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

public class Predictions {

	public static void main(String[] args) {
		// TODO Auto-generated method stub

		if(args.length == 0) {
			System.out.println("Programa que recoge el ARFF test y el modelo óptimo y realiza las predicciones");
    		System.out.println("@pre El archivo de arff no esté vacío, la clase es el último del atributo y el modelo sea el más óptimo");
    		System.out.println("@post El resultado de las predicciones");
    		System.out.println("@param Ruta del fichero ARFF train");
    		System.out.println("@param Ruta del fichero ARFF test");
    		System.out.println("@param Ruta del modelo");
    		System.out.println("@param Ruta dónde guardar el resultado TXT de las predicciones");
		}else {
			try {
				DataSource sourceTrain = new DataSource(args[0]);
				Instances train = sourceTrain.getDataSet();
				train.setClassIndex(train.numAttributes()-1);
				
				DataSource sourceTest = new DataSource(args[1]);
				Instances test = sourceTest.getDataSet();
				test.setClassIndex(test.numAttributes()-1);
				
				String rutaModelo = args[2];
				String rutaResultado = args[3];		
				
				predicciones(train, test, rutaModelo, rutaResultado);
				
			}catch (Exception e) {
				e.printStackTrace();
			}
			
		}
	}
	
	private static void predicciones(Instances train, Instances test, String modelo, String resultado) throws Exception {
	
		DecimalFormat formato = new DecimalFormat("0.00");
		
		BayesNet bayesNet = (BayesNet) SerializationHelper.read(modelo);
		bayesNet.buildClassifier(train);
		
		Evaluation ev = new Evaluation(test);
		ev.evaluateModel(bayesNet, test);
				
		ArrayList<Prediction> predicciones = ev.predictions();
		
		if(predicciones.size() >0) {
			StringBuilder sb = new StringBuilder();
			String res="";
			
			for(int i=0;i<predicciones.size();i++) {
				NominalPrediction np = (NominalPrediction) predicciones.get(i);
				Attribute atrib = test.attribute(test.classIndex());
				
				int predicho = (int) np.predicted();
				int real = (int) np.actual();
				String error ="";
				
				double marginCorrect =0;
				double marginIncorrect=0;
								
				if(predicho !=real) {
					error = "+";
					marginCorrect = np.margin()*(-1);
					marginIncorrect = 1-marginCorrect;
				}else {
					marginCorrect = np.margin();
					marginIncorrect = 1- marginCorrect;
				}
				
				res = i 
						+ " | Predicho: " + atrib.value(predicho)
						+ " | Real : " + atrib.value(real)
						+ " | Error : " + error 
						+ " | Margin: " + formato.format(marginCorrect)
						+ " | " + formato.format(marginIncorrect)
						+ " | Distribución : " + formato.format(np.distribution()[predicho])
						+ " | " + formato.format(np.distribution()[real])
						+"\n";
				sb.append(res);
			}
			
			BufferedWriter bw = new BufferedWriter(new FileWriter(resultado));
			bw.append(sb.toString());
			bw.newLine();
			bw.close();
			
		}
	}

}
