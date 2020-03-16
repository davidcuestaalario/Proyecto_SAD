package Proyecto_Preguntas;

public class Main 
{
	public static void main(String[] args) 
    {
		String ruta = "DATA2";
		PreProcesado preprocesado = new PreProcesado( ruta );
		
		// FICHERO
		preprocesado.getRawARFF( "train" );
		preprocesado.getRawARFF( "test_unk" );
		
		// FICHERO , BOW-TFIDF , Sparse-NonSparse
		preprocesado.getWordVector( "train" , "BoW"   , "Sparse" );
		preprocesado.getWordVector( "train" , "TFIDF" , "Sparse" );
		preprocesado.getWordVector( "train" , "BoW"   , "NonSparse" );
		preprocesado.getWordVector( "train" , "TFIDF" , "NonSparse" );
		
		// BOW-TFIDF , Sparse-NonSparse
		preprocesado.filtradoAtributos("BoW"   , "Sparse");
		preprocesado.filtradoAtributos("TFIDF" , "Sparse");
		preprocesado.filtradoAtributos("BoW"   , "NonSparse");
		preprocesado.filtradoAtributos("TFIDF" , "NonSparse");
		
    }
}
