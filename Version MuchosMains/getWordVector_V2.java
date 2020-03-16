package Proyecto;

import java.io.BufferedWriter;
import java.io.FileWriter;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Reorder;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.unsupervised.instance.NonSparseToSparse;
import weka.filters.unsupervised.instance.SparseToNonSparse;

public class getWordVector_V2 
{
	public static void main(String[] args) 
    {
		System.out.println(" ------------------- RaW To WordVector ------------------- ");
		System.out.println("Programa que aplica la vectorizacion para cada uno de los textos");
		System.out.println("");
		System.out.println("@pre Se han transformado los textos a un fichero arff que tiene la clase en el indice cero y el string en el indice uno.");
		System.out.println("@post La clase a predecir pasa ha estar en la ultima posicion.");
		System.out.println("");
		System.out.println("@param Ruta del fichero RAW en formato arff");
		System.out.println("@param Ruta donde guardar el fichero BOW en formato arff");
		System.out.println("@param Ruta donde guardar el dicionario en formato txt");
		System.out.println("@param Indicar del tipo de WordVector BoW o TF-IDF");
		System.out.println("@param Indicar el tipo de compresion Sparse o NonSparse");
		System.out.println("");
		
		// String name = "test_unk";
		String name = "train";
		boolean sparse = true;
    	boolean bow = false;
    	
		String config = "";
		
		if ( bow ) { config += "BoW_"; }
    	else { config += "TFIDF_"; }
    	
    	if ( sparse ) { config += "Sparse"; }
    	else { config += "NonSparse"; }
    		    	
    	String rawarff = "DATA/" + name + "RaW.arff";
    	String bowarff = "DATA/" + name + config + ".arff";   	
    	
		if(args.length == 3)
		{
			rawarff = args[0];
	    	bowarff = args[1];
	    	name = args[2];
	    	config = args[3];
	    	if( name == "Sparse" ) { sparse = false; };
	    	if( config == "bow" ) { bow = false; }
		}
		
		try
	    {
			System.out.println("Iniciando la conversion RAW -> BOW...");
			
			// CARGAMOS EL CONJUNTO DE DATOS DEL RAW
			Instances data = new DataSource( rawarff ).getDataSet();

			// APLICAMOS EL FILTRO StringToWordVector
	    	StringToWordVector filtroBow = new StringToWordVector();
	    	filtroBow.setInputFormat(data);
	    	filtroBow.setOutputWordCounts( !bow );  
	    	filtroBow.setIDFTransform( !bow );
	    	filtroBow.setTFTransform( !bow );
	    	filtroBow.setLowerCaseTokens(true);
	    	filtroBow.setOutputWordCounts(true);
	    	filtroBow.setOutputWordCounts(true);		    	
	    	data = Filter.useFilter(data, filtroBow);
	    	System.out.println("Filtro StringToWordVector aplicado...");
	    	
	    	// APLICAMOS EL FILTRO NonSparseToSparse
	    	if ( !sparse ) 
	    	{
	    		NonSparseToSparse filtroSparse = new NonSparseToSparse();
           	 	filtroSparse.setInputFormat(data);
           	 	data = Filter.useFilter(data, filtroSparse);
           	 	System.out.println("Filtro NonSparseToSparsede aplicado...");
	    	}
	    	else
	    	{
	    		SparseToNonSparse filtroSparse = new SparseToNonSparse();
           	 	filtroSparse.setInputFormat(data);
           	 	data = Filter.useFilter(data, filtroSparse);
           	 	System.out.println("Filtro NonSparseToSparsede aplicado...");
	    	}
	    	
	    	// MOVEMOS LA CLASE A PREDECIR AL FINAL
			Reorder filtroR = new Reorder();
			filtroR.setAttributeIndices("2-last,1");
			filtroR.setInputFormat(data);
			System.out.println("Clase a predecir movida al ultimo atributo...");
							           	 	
			// SETTEAR NOMBRE DE LA RELACION
        	data.setRelationName("preguntas_" + name + "_" + config);
        	System.out.println("Nombre de la relacion establecida...");
        	           	 
	        // GUARDAR EL CONJUNTO DE DATOS EN UN ARFF 	
       	 	BufferedWriter writer = new BufferedWriter(new FileWriter( bowarff ));
       	 	writer.write(data.toString());
       	 	writer.flush();
       	 	writer.close();
       	 	System.out.println("Conversion finalizada...");
       	 	
       	 	
	    }    
		catch (Exception e) { e.printStackTrace(); }
    }
}