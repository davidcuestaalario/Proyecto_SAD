package Proyecto;

import java.io.BufferedWriter;
import java.io.FileWriter;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.Remove;

public class FiltradoAtributos 
{
	public static void main(String[] args) 
    {
		System.out.println(" ------------------- Filtrado de atributos ------------------- ");
		System.out.println("Programa que obtiene filtra los atributos que no correlan con la clase y los que correlan entre si");
		System.out.println("Permitiendo eliminar aquellos atributos que describen la clase de forma similar, ");
		System.out.println("y aquellos atributos que no aportan información útil para diferenciar la clase"); 
		System.out.println("");
		System.out.println("@pre El fichero de train esta en formato BoW o TFIDF con extension arff y el fichero de test en formato RaW con extension arff");
		System.out.println("@post Tanto el fichero de train como el de test tendran los mismos atributos en el mismo orden.");
		System.out.println("");
		System.out.println("@param Ruta del fichero train BoW o TFIDF");
		System.out.println("@param Ruta del fichero test_unk RaW");
		System.out.println("@param Ruta donde guardar el fichero train filtrado");
		System.out.println("@param Ruta donde guardar el fichero test_unk filtrado");
		
		//String bow = "BoW";
		String bow = "TFIDF";
		String sparse = "Sparse";
		//String sparse = "NonSparse";
		
		String train_original = "DATA/train_" + bow + "_" + sparse + ".arff" ;
		String test_original = "DATA/test_unk_RaW.arff";
		String train_resultado = "DATA/train_" + bow + "_" + sparse + "_Filtrados.arff";
		String test_resultado = "DATA/test_unk_" + bow + "_" + sparse + "_Filtrados.arff";
		System.out.println(train_original);
		if( args.length == 4 )
		{
			train_original = args[0];
			test_original = args[1];
			train_resultado = args[2];
			test_resultado = args[3];
		}
		
		try
		{						
			// FILTRAMOS LOS DATOS DE ENTRENAMIENTO
			System.out.println("Iniciando el filtrado de los datos de entrenamiento...");
			Instances dataTrain = new DataSource( train_original ).getDataSet();
			
			// APLICAMOS LA SELECCION DE ATRIBUTOS AL TRAIN EN FUNCION DE LA CORRELACION	        
	        AttributeSelection filtro = new AttributeSelection();
			CfsSubsetEval cfs = new CfsSubsetEval();
			BestFirst search = new BestFirst();
				
			filtro.setSearch(search);
			filtro.setEvaluator(cfs);
			filtro.setInputFormat( dataTrain );
			
			dataTrain = filtro.useFilter( dataTrain , filtro);
			System.out.println("Datos de entrenamiento filtrados...");
			
			// SETTEAR NOMBRE DE LA RELACION
			dataTrain.setRelationName("preguntas_" + bow + "_" + sparse + "_Filtrados");
        	System.out.println("Nombre de la relacion establecida...");
        	           	 
	        // GUARDAR EL CONJUNTO DE DATOS EN UN ARFF 	
       	 	BufferedWriter trainWriter = new BufferedWriter( new FileWriter( train_resultado ) );
       	 	trainWriter.write( dataTrain.toString() );
       	 	trainWriter.flush();
       	 	trainWriter.close();
       	 	System.out.println("Filtrado finalizado...");
			
       	 	
       	 	
		}
		catch(Exception e) { e.printStackTrace(); }
	}
}