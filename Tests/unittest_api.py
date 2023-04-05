import unittest
import requests
import testdata

class TestAPI(unittest.TestCase):

    def setUp(self):
        self.url = 'https://nephroapi.azurewebsites.net/api/nephro_ai'

    # Check an actual CT scan report
    def test_post_m0(self):
        base64 = testdata.generator(0)
        data = {"img": base64}
        response = requests.post(self.url, json=data)
        self.assertEqual(response.status_code, 200)
        
    # Check with file size (Minimum)
    def test_post_m1(self):
        base64 = testdata.generator(1)
        data = {"img": base64}
        response = requests.post(self.url, json=data)
        self.assertEqual(response.text, 'Image size is too small!')
        self.assertEqual(response.status_code, 400)

    # Check with an empty body
    def test_post_m2(self):
        data = {'img': ''}
        response = requests.post(self.url, json=data)
        self.assertEqual(response.text, 'Empty body, please make sure that your request is valid!')
        self.assertEqual(response.status_code, 404)
        
    # Check with an invalid file type
    def test_post_m3(self):
        base64 = testdata.generator(2)
        data = {'img': base64}
        response = requests.post(self.url, json=data)
        self.assertEqual(response.status_code, 400)
        
    # Check with file size (Maximum)
    def test_post_m3(self):
        base64 = testdata.generator(5)
        data = {'img': base64}
        response = requests.post(self.url, json=data)
        self.assertEqual(response.status_code, 400)
    
     # Check with file size (check with non-CT files)
    def test_post_m3(self):
        base64 = testdata.generator(4)
        data = {'img': base64}
        response = requests.post(self.url, json=data)
        self.assertEqual(response.status_code, 400)
    


if __name__ == '__main__':
    unittest.main()