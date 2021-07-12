# class used to find the max microscopic cross-section value
class MaxSigma(dict):
    def __getitem__(self, parent_product_mt):
        """
        key is provided in the format of Pt206-Pt207-MT=(102,5).
        we will then return  up the following:
        
        max([max_sigma["Pt206-Pt207-MT=102"], max_sigma["Pt206-Pt207-MT=5"]])
        """
        parent_product_, mts = parent_product_mt.split("=")
        results = []
        for mt in mts.strip("()").split(","):
            results.append(super(MaxSigma, self).__getitem__(parent_product_ +"="+ mt))
        return max(results)