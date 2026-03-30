package check

import (
	"strings"
	"testing"
)

func TestExtractGoSignatures(t *testing.T) {
	content := `package auth

import "testing"

func TestLogin(t *testing.T) {
	client := NewClient()
	server := httptest.NewServer(handler)
	defer server.Close()

	body := strings.NewReader(` + "`" + `{"user":"admin","pass":"secret"}` + "`" + `)
	resp, err := client.Post(server.URL+"/login", "application/json", body)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		t.Errorf("expected 200, got %d", resp.StatusCode)
	}

	var result map[string]string
	json.NewDecoder(resp.Body).Decode(&result)
	if result["token"] == "" {
		t.Error("expected non-empty token")
	}
}

func TestLoginSubcases(t *testing.T) {
	t.Run("valid credentials", func(t *testing.T) {
		// lots of setup
		token := doLogin(t, "admin", "pass")
		if token == "" {
			t.Fatal("empty token")
		}
	})

	t.Run("invalid password", func(t *testing.T) {
		resp := tryLogin("admin", "wrong")
		if resp.StatusCode != 401 {
			t.Errorf("expected 401, got %d", resp.StatusCode)
		}
	})
}

func helperSetup(t *testing.T) *Client {
	t.Helper()
	return &Client{baseURL: "http://localhost"}
}
`

	sig := extractTestSignatures("auth_test.go", content)

	// Should keep function signatures
	if !strings.Contains(sig, "func TestLogin(t *testing.T)") {
		t.Error("should keep TestLogin signature")
	}
	if !strings.Contains(sig, "func TestLoginSubcases(t *testing.T)") {
		t.Error("should keep TestLoginSubcases signature")
	}

	// Should keep subtests
	if !strings.Contains(sig, `t.Run("valid credentials"`) {
		t.Error("should keep t.Run calls")
	}
	if !strings.Contains(sig, `t.Run("invalid password"`) {
		t.Error("should keep t.Run calls")
	}

	// Should keep assertions
	if !strings.Contains(sig, "t.Fatal(err)") {
		t.Error("should keep t.Fatal")
	}
	if !strings.Contains(sig, "t.Errorf") {
		t.Error("should keep t.Errorf")
	}
	if !strings.Contains(sig, `t.Error("expected non-empty token")`) {
		t.Error("should keep t.Error")
	}

	// Should keep helper marker
	if !strings.Contains(sig, "t.Helper()") {
		t.Error("should keep t.Helper()")
	}

	// Should NOT keep setup/boilerplate
	if strings.Contains(sig, "NewClient()") {
		t.Error("should not keep setup code")
	}
	if strings.Contains(sig, "httptest.NewServer") {
		t.Error("should not keep httptest setup")
	}
	if strings.Contains(sig, "json.NewDecoder") {
		t.Error("should not keep json decoding")
	}
	if strings.Contains(sig, "defer") {
		t.Error("should not keep defer statements")
	}

	// Should be significantly smaller
	ratio := float64(len(sig)) / float64(len(content))
	if ratio > 0.5 {
		t.Errorf("expected >50%% reduction, got %.0f%% of original", ratio*100)
	}
}

func TestExtractRustSignatures(t *testing.T) {
	content := `use crate::backend::CudaBackend;

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_backend() -> CudaBackend {
        let config = Config::default();
        CudaBackend::new(&config).unwrap()
    }

    #[test]
    fn test_rmsnorm_forward() {
        let backend = setup_backend();
        let input = create_test_tensor(vec![1, 4096]);
        let weight = create_test_tensor(vec![4096]);
        let output = backend.rmsnorm(&input, &weight, 1e-5).unwrap();
        let reference = load_reference("rmsnorm_output.bin");
        assert_eq!(output.shape(), reference.shape());
        assert!(tensors_close(&output, &reference, 1e-3, 1e-3));
    }

    #[test]
    fn test_rmsnorm_zero_input() {
        let backend = setup_backend();
        let zeros = Tensor::zeros(vec![1, 4096]);
        let weight = Tensor::ones(vec![4096]);
        let output = backend.rmsnorm(&zeros, &weight, 1e-5).unwrap();
        assert!(output.all_close_to(0.0, 1e-6));
    }
}
`

	sig := extractTestSignatures("rmsnorm_test.rs", content)

	if !strings.Contains(sig, "#[cfg(test)]") {
		t.Error("should keep #[cfg(test)]")
	}
	if !strings.Contains(sig, "#[test]") {
		t.Error("should keep #[test]")
	}
	if !strings.Contains(sig, "fn test_rmsnorm_forward()") {
		t.Error("should keep test fn signature")
	}
	if !strings.Contains(sig, "fn test_rmsnorm_zero_input()") {
		t.Error("should keep test fn signature")
	}
	if !strings.Contains(sig, "assert_eq!") {
		t.Error("should keep assert_eq!")
	}
	if !strings.Contains(sig, "assert!") {
		t.Error("should keep assert!")
	}
	if !strings.Contains(sig, ".unwrap()") {
		t.Error("should keep .unwrap() lines")
	}

	// Should not keep plain setup
	if strings.Contains(sig, "Config::default()") {
		t.Error("should not keep setup code")
	}
}

func TestExtractPythonSignatures(t *testing.T) {
	content := `import pytest
from myapp import login

class TestAuth:
    def setup_method(self):
        self.client = Client()

    def test_login_success(self):
        response = self.client.post("/login", data={"user": "admin"})
        assert response.status_code == 200
        assert "token" in response.json()

    def test_login_failure(self):
        response = self.client.post("/login", data={"user": "bad"})
        self.assertEqual(response.status_code, 401)

def test_standalone():
    with pytest.raises(ValueError):
        login(None, None)
`

	sig := extractTestSignatures("test_auth.py", content)

	if !strings.Contains(sig, "class TestAuth:") {
		t.Error("should keep test class")
	}
	if !strings.Contains(sig, "def test_login_success") {
		t.Error("should keep test method")
	}
	if !strings.Contains(sig, "def test_standalone") {
		t.Error("should keep test function")
	}
	if !strings.Contains(sig, "assert response.status_code == 200") {
		t.Error("should keep assert")
	}
	if !strings.Contains(sig, "pytest.raises") {
		t.Error("should keep pytest.raises")
	}

	if strings.Contains(sig, "self.client = Client()") {
		t.Error("should not keep setup code")
	}
}

func TestExtractUnknownLanguageFallback(t *testing.T) {
	content := "some test content in unknown language"
	sig := extractTestSignatures("test.xyz", content)
	if sig != content {
		t.Error("should return full content for unknown file types")
	}
}

func TestExtractEmptyFileReturnsContent(t *testing.T) {
	// If extraction produces nothing (no matching lines), return full content
	content := `package main
// this file has no test patterns at all
var x = 1
`
	sig := extractTestSignatures("weird_test.go", content)
	if sig != content {
		t.Error("should fall back to full content when no signatures found")
	}
}

func TestExtractJSSignatures(t *testing.T) {
	content := `const { login } = require('./auth');

describe('Auth', () => {
  let client;

  beforeEach(() => {
    client = new Client();
  });

  it('should login with valid creds', async () => {
    const result = await client.login('admin', 'pass');
    expect(result.token).toBeDefined();
    expect(result.status).toBe(200);
  });

  test('rejects bad password', () => {
    expect(() => login('admin', 'wrong')).toThrow();
  });
});
`

	sig := extractTestSignatures("auth.test.js", content)

	if !strings.Contains(sig, "describe(") {
		t.Error("should keep describe blocks")
	}
	if !strings.Contains(sig, "it(") {
		t.Error("should keep it blocks")
	}
	if !strings.Contains(sig, "test(") {
		t.Error("should keep test blocks")
	}
	if !strings.Contains(sig, "expect(") {
		t.Error("should keep expect assertions")
	}
	if !strings.Contains(sig, "beforeEach(") {
		t.Error("should keep beforeEach")
	}

	if strings.Contains(sig, "new Client()") {
		t.Error("should not keep setup internals")
	}
}
