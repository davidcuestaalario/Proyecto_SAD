
import java.io.BufferedWriter;
import java.io.FileWriter;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

public class Predictions {

	public static void main(String[] args) {

		if(args.length == 0) {
			System.out.println("Programa que recoge el ARFF test, el modelo óptimo y los archivos dev ARFF y realiza las predicciones");
    		System.out.println("@pre Los archivos de arffs no estén vacíos, la clase es el último del atributo y el modelo sea el más óptimo");
    		System.out.println("@post El resultado de las predicciones en un fichero de texto");
    		System.out.println("@param Ruta del fichero ARFF dev 70% ");
    		System.out.println("@param Ruta del fichero ARFF dev 30% ");
    		System.out.println("@param Ruta del modelo Bayes Network");
    		System.out.println("@param Ruta del modelo Naive Bayes Multinomial");
    		System.out.println("@param Ruta dónde guardar el resultado TXT de las predicciones Bayes Network");
    		System.out.println("@param Ruta dónde guardar el resultado TXT de las predicciones Naive Bayes Multinomial");
    	
		}else if(args.length == 6) {
			try {
				String fichDev70 = args[0];
				String fichDev30 = args[1];
				
				DataSource sourceDev70 = new DataSource(fichDev70);
				Instances dev70 = sourceDev70.getDataSet();
				if(esFicheroFiltrado(fichDev70)) {
					dev70.setClassIndex(dev70.numAttributes()-1);
				}else {
					dev70.setClassIndex(0);
				}
				
				DataSource sourceDev30 = new DataSource(fichDev30);
				Instances dev30 = sourceDev30.getDataSet();
				if(esFicheroFiltrado(fichDev70)) {
					dev30.setClassIndex(dev70.numAttributes()-1);
				}else {
					dev30.setClassIndex(0);
				}
				
				String rutaModeloBayesNet = args[2];
				String rutaModeloMultinomial = args[3];
				String rutaResultadoBayesNetwork = args[4];		
				String rutaResultadoMultinomial = args[5];
				
				System.out.println("Realizando predicciones de Bayes Network...");
				prediccionesBayesNetwork(dev70,dev30, rutaModeloBayesNet, rutaResultadoBayesNetwork);
				System.out.println("Predicciones Bayes Network realizadas");
				
				System.out.println("Realizando predicciones de Naive Bayes Multinomial...");
				prediccionesMultinomial(dev70,dev30, rutaModeloMultinomial, rutaResultadoMultinomial);
				System.out.println("Predicciones Naive Bayes Multinomial realizadas");
				
			}catch (Exception e) {
				e.printStackTrace();
			}
		}
	}
	
	private static boolean esFicheroFiltrado(String fichero) {
		if(fichero.contains("Filtrados")) {
			return true;
		}
		return false;
	}
	
	private static void prediccionesBayesNetwork(Instances dev70,Instances dev30, String modelo, String resultado) throws Exception {
			
		BayesNet bayesNet = (BayesNet) SerializationHelper.read(modelo);
		bayesNet.buildClassifier(dev70);
		Evaluation ev = new Evaluation(dev70);
		
		StringBuilder sb = new StringBuilder();
		
		double predicho =0;
		double real =0;
		
		for(int i = 0; i < dev30.numInstances(); i++){
			predicho = ev.evaluateModelOnceAndRecordPrediction(bayesNet, dev30.instance(i));
			real = dev30.instance(i).classValue();
			String error ="";
			
			if(predicho !=real) {
				error = "+";
			}
			sb.append("Instance: "+ i
					+ " | Predicted: " + dev30.classAttribute().value((int) predicho)
					+ " | Real: " + dev30.classAttribute().value((int) real)
					+ " | Error: " + error
					+ "\n");
		}
		
		BufferedWriter bw = new BufferedWriter(new FileWriter(resultado));
		bw.append("PREDICCIONES BAYES NETWORK\n");
		bw.newLine();
		bw.append(sb.toString());
		bw.close();
		
	}
	
	private static void prediccionesMultinomial(Instances dev70,Instances dev30, String modelo, String resultado) throws Exception {
		
		NaiveBayesMultinomial baseline = (NaiveBayesMultinomial) SerializationHelper.read(modelo);
		baseline.buildClassifier(dev70);
		
		Evaluation ev = new Evaluation(dev70);
		
		StringBuilder sb = new StringBuilder();
		
		double predicho =0;
		double real =0;
		for(int i = 0; i < dev30.numInstances(); i++){
			predicho = ev.evaluateModelOnceAndRecordPrediction(baseline, dev30.instance(i));
			real = dev30.instance(i).classValue();
			String error ="";
			
			if(predicho !=real) {
				error = "+";
			}
			sb.append("Instance: "+ i
					+ " | Predicted: " + dev30.classAttribute().value((int) predicho)
					+ " | Real: " + dev30.classAttribute().value((int) real)
					+ " | Error: " + error
					+ "\n");
		}
		
		System.out.println(ev.toSummaryString());

		BufferedWriter bw = new BufferedWriter(new FileWriter(resultado));
		bw.append("PREDICCIONES NAIVE BAYES MULTINOMIAL\n");
		bw.newLine();
		bw.append(sb.toString());
		bw.close();
		
	}

}
